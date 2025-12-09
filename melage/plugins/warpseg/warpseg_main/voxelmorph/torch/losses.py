import math
from torch.autograd import Variable
from math import exp
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleMSE(nn.Module):
    """Multi-scale Mean Squared Error loss using average pooling."""
    def __init__(self, scales=[1, 2, 4]):
        super(MultiScaleMSE, self).__init__()
        self.scales = scales
        self.mse = nn.MSELoss()

    def forward(self, I, J):
        total_loss = 0.0
        for scale in self.scales:
            if scale > 1:
                I_scaled = F.avg_pool3d(I, kernel_size=scale, stride=scale, padding=0)
                J_scaled = F.avg_pool3d(J, kernel_size=scale, stride=scale, padding=0)
            else:
                I_scaled = I
                J_scaled = J

            total_loss += self.mse(I_scaled, J_scaled)

        return total_loss / len(self.scales)



def calculate_jacobian_determinant_3d(flow):
    """
    Calculates the Jacobian determinant of a 3D deformation field.

    Args:
        flow (torch.Tensor): The deformation field, shape (B, 3, D, H, W).
                             Displacements are in (x, y, z) order.

    Returns:
        torch.Tensor: The Jacobian determinant map, shape (B, 1, D, H, W).
    """
    # Get spatial dimensions
    B, _, D, H, W = flow.shape

    # Create an identity grid
    grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(D, device=flow.device),
                                            torch.arange(H, device=flow.device),
                                            torch.arange(W, device=flow.device),
                                            indexing='ij')
    grid = torch.stack((grid_x, grid_y, grid_z), dim=0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape: (B, 3, D, H, W)

    # Full transformation map
    transform_map = grid + flow

    # Calculate spatial gradients (central differences)
    # Gradient along x (W dimension)
    diff_x = transform_map[..., 1:] - transform_map[..., :-1]
    grad_x = torch.cat([diff_x, diff_x[..., -1:]], dim=-1)

    # Gradient along y (H dimension)
    diff_y = transform_map[..., 1:, :] - transform_map[..., :-1, :]
    grad_y = torch.cat([diff_y, diff_y[..., -1:, :]], dim=-2)

    # Gradient along z (D dimension)
    diff_z = transform_map[:, :, 1:, :, :] - transform_map[:, :, :-1, :, :]
    grad_z = torch.cat([diff_z, diff_z[:, :, -1:, :, :]], dim=-3)
    # Unpack gradients
    J_xx, J_yx, J_zx = grad_x[:, 0, ...], grad_x[:, 1, ...], grad_x[:, 2, ...]
    J_xy, J_yy, J_zy = grad_y[:, 0, ...], grad_y[:, 1, ...], grad_y[:, 2, ...]
    J_xz, J_yz, J_zz = grad_z[:, 0, ...], grad_z[:, 1, ...], grad_z[:, 2, ...]

    # Compute the 3x3 determinant
    determinant = (J_xx * (J_yy * J_zz - J_zy * J_yz) -
                   J_xy * (J_yx * J_zz - J_zx * J_yz) +
                   J_xz * (J_yx * J_zy - J_zx * J_yy))

    return determinant.unsqueeze(1)



def compute_jacobian_determinant(disp):
    """
    Compute the Jacobian determinant of a 3D displacement field.
    Args:
        disp (torch.Tensor): Displacement field of shape (B, 3, D, H, W)
    Returns:
        jac_det (torch.Tensor): Jacobian determinant of shape (B, D, H, W)
    """
    B, C, D, H, W = disp.shape
    grad = torch.gradient(disp, dim=(2, 3, 4))  # Gradient along each spatial axis

    # Construct Jacobian matrix components
    dx = grad[0]  # (B, 3, D, H, W)
    dy = grad[1]  # (B, 3, D, H, W)
    dz = grad[2]  # (B, 3, D, H, W)

    # Assemble the Jacobian matrix
    jacobian = torch.zeros((B, D, H, W, 3, 3), device=disp.device)
    jacobian[..., 0, 0] = dx[:, 0] + 1  # ∂φx/∂x + identity
    jacobian[..., 0, 1] = dy[:, 0]
    jacobian[..., 0, 2] = dz[:, 0]
    jacobian[..., 1, 0] = dx[:, 1]
    jacobian[..., 1, 1] = dy[:, 1] + 1  # ∂φy/∂y + identity
    jacobian[..., 1, 2] = dz[:, 1]
    jacobian[..., 2, 0] = dx[:, 2]
    jacobian[..., 2, 1] = dy[:, 2]
    jacobian[..., 2, 2] = dz[:, 2] + 1  # ∂φz/∂z + identity

    # Compute the determinant of the Jacobian matrix
    jac_det = torch.det(jacobian)



    return jac_det

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))/(np.sqrt(2*np.pi)*sigma) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim3D(img1, img2, window, window_size, channel, L=1):

    mux = F.conv3d(img1, window, padding=window_size // 2, groups=channel) #Overall Mean Luminance im1
    muy = F.conv3d(img2, window, padding=window_size // 2, groups=channel)#Overall Mean Luminance im2
    mux_sq = mux.pow(2)
    muy_sq = muy.pow(2)
    # Constants for SSIM calculation
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2


    mux_muy = mux * muy

    sigmax_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mux_sq
    sigmay_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - muy_sq
    sigmaxy = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mux_muy

    ssim_map = ((2 * mux_muy + C1) * (2 * sigmaxy + C2)) / ((mux_sq + muy_sq + C1) * (sigmax_sq + sigmay_sq + C2))


    return ssim_map



def _ssim3D_new(img1, img2, window, window_size, channel, L=1):
    device = img1.device
    window = window.to(device)
    mux = F.conv3d(img1, window, padding=window_size // 2, groups=channel) #Overall Mean Luminance im1
    #device = img2.device
    img2 = img2.to(device)

    muy = F.conv3d(img2, window, padding=window_size // 2, groups=channel)#Overall Mean Luminance im2
    mux_sq = mux.pow(2)
    muy_sq = muy.pow(2)
    # Constants for SSIM calculation
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2


    mux_muy = mux * muy

    sigmax_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mux_sq
    sigmay_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - muy_sq
    sigmaxy = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mux_muy
    #
    #
    #ssim_map = ((2 * mux_muy + C1) * (2 * sigmaxy + C2)) / ((mux_sq + muy_sq + C1) * (sigmax_sq + sigmay_sq + C2))
    # Compute texture and structure maps
    structure_map= (2 * sigmaxy+C2) / (sigmax_sq + sigmay_sq + C2)
    texture_map= (2 * mux_muy + C1) / (mux_sq + muy_sq + C1)
    return texture_map, structure_map


def match_intensity(template, base):
    base = sitk.GetImageFromArray(base)
    tm = sitk.GetImageFromArray(template)
    matched_atlas = sitk.HistogramMatching(tm, base)
    matched_atlas = sitk.GetArrayFromImage(matched_atlas)

    return matched_atlas


class SSIM3D_DIST(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, device='cpu'):
        super(SSIM3D_DIST, self).__init__()
        self.window_size = window_size
        self.device = device
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)
        #self.alpha = nn.Parameter(torch.ones(1))
        #self.beta = nn.Parameter(torch.ones(1))
        #self.gamma = nn.Parameter(torch.ones(1))
        self.eps = 1e-7
        self.weight_raw = torch.nn.Parameter(torch.ones(0))



    def forward(self, gt, pred, uncertain_score=None, map=False):
        (batch_size, channel, _, _, _) = gt.size()

        if channel == self.channel and self.window.data.type() == gt.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if gt.is_cuda:
                window = window.cuda(gt.get_device())
            window = window.type_as(gt)

            self.window = window
            self.channel = channel

        #texture_map, structure_map = _ssim3D_new(gt, pred, window, self.window_size, channel)
        ssim_map = _ssim3D(gt, pred, window, self.window_size, channel)
        #ssim_map = 0.8*texture_map+0.2*structure_map

        if map:
            loss_ssim = (1-ssim_map)
        else:
            loss_ssim = (1-ssim_map).mean()
        return loss_ssim


        #weight = torch.abs(gt - pred).mean(dim=(1, 2, 3, 4), keepdim=True)  # Adaptive weight
        #weight = (weight - weight.min()) / (weight.max() - weight.min() + self.eps)  # Normalize
        weight=0.5
        loss_ssim = 0
        for i in range(batch_size):
            #a1 = gt.detach().squeeze().cpu().numpy()[i, ...]
            #b1 = pred.detach().squeeze().cpu().numpy()[i, ...]

            #c = match_intensity(a1, b1)
            #weight = abs(c - a1)
            #weight = (weight - weight.min()) / ((weight.max() - weight.min()) + self.eps)
            #weight = torch.from_numpy(weight).to(self.device)
            #weight = 0.5

            ssim_map = (weight[i,...]) * (texture_map[i, ...]) + (1 - weight) * (structure_map[i, ...])
            # ssim_map = (weight) * ((Lumin_ssim*Contrast_ssim)[i,...]) + (1-weight) * (structure_ssim[i,...])
            # ssim_map = (weight)*(Lumin_ssim*Contrast_ssim) + (1-weight)*structure_ssim
            if uncertain_score is not None:
                
                loss_ssim += ((1-ssim_map)*uncertain_score[i].unsqueeze(0)).mean()
            #loss_ssim += 1- ssim_map
        # ssim_map = (texture_map) + (structure_map)
        # texture_map, structure_map = _ssim3D(im1, im2, window, self.window_size, channel, L=1)
        # loss_ssim += (1-ssim_map.mean([1,2,3,4])).mean()

        return loss_ssim / batch_size


    def forward_old(self, gt, pred):
        (batch_size, channel, _, _, _) = gt.size()

        if channel == self.channel and self.window.data.type() == gt.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if gt.is_cuda:
                window = window.cuda(gt.get_device())
            window = window.type_as(gt)

            self.window = window
            self.channel = channel

        # Apply softmax and scale to ensure alpha + beta = 1
        #alpha_beta = torch.cat((self.alpha, self.beta), dim=0)
        #weight = self.weight_raw.exp()
        #weight = weight / weight.sum(0, keepdim=True).clamp(min=self.eps)
        weight = [0.5, 0.5]
        #alpha_beta_softmax = F.softmax(alpha_beta, dim=0)
        #self.alpha.data = alpha_beta_softmax[0]
        #self.beta.data = alpha_beta_softmax[1]
        # https://arxiv.org/pdf/2004.07728

        #Lumin_ssim, Contrast_ssim, structure_ssim = _Localssim3D(gt, pred, window, self.window_size, channel)
        #ssim_map = _ssim3D(gt, pred, window, self.window_size, channel)
        #return  (1-ssim_map.mean([1,2,3,4])).sum()

        texture_map, structure_map = _ssim3D_new(gt, pred, window, self.window_size, channel)
        loss_ssim = 0
        for i in range(batch_size):
            a1 = gt.detach().squeeze().cpu().numpy()[i, ...]
            b1 = pred.detach().squeeze().cpu().numpy()[i, ...]

            c = match_intensity(a1, b1)
            weight = abs(c - a1)
            weight = (weight - weight.min()) / ((weight.max() - weight.min()) + self.eps)
            weight = torch.from_numpy(weight).to(self.device)

            ssim_map = (weight) * (texture_map[i, ...]) + (1 - weight) * (structure_map[i, ...])
            # ssim_map = (weight) * ((Lumin_ssim*Contrast_ssim)[i,...]) + (1-weight) * (structure_ssim[i,...])
            # ssim_map = (weight)*(Lumin_ssim*Contrast_ssim) + (1-weight)*structure_ssim
            loss_ssim += (1 - ssim_map.mean())
            #loss_ssim += 1- ssim_map
        # ssim_map = (texture_map) + (structure_map)
        # texture_map, structure_map = _ssim3D(im1, im2, window, self.window_size, channel, L=1)
        # loss_ssim += (1-ssim_map.mean([1,2,3,4])).mean()

        return loss_ssim / batch_size

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, uncertain_score=None):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win
        b,c,h,w,d = Ii.shape
        # compute filters
        sum_filt = torch.ones([c,1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding, groups=c)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding, groups=c)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding, groups=c)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding, groups=c)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding, groups=c)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Clamp variances to be non-negative
        I_var = torch.clamp(I_var, min=1e-5)
        J_var = torch.clamp(J_var, min=1e-5)

        cc = cross * cross / (I_var * J_var + 1e-5)

        # Avoid NaN by replacing any NaN values with zero (optional)
        cc = torch.nan_to_num(cc, nan=0.0)
        if uncertain_score is not None:
            return -(0.1*(cc*uncertain_score).sum(1).mean()+(cc).sum(1).mean()) #sum over channel
        else:
            return -cc.sum(1).mean() #sum over channel


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice





def total_variation(prob_pred):

    tv_d = torch.pow(prob_pred[:, :, 1:, :, :] - prob_pred[:, :, :-1, :, :],2).mean()
    tv_h = torch.pow(prob_pred[:, :, :, 1:, :] - prob_pred[:, :, :, :-1, :],2).mean()
    tv_w = torch.pow(prob_pred[:, :, :, :, 1:] - prob_pred[:, :, :, :, :-1],2).mean()
    return tv_d + tv_h + tv_w

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

