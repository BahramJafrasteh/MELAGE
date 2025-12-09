from nibabel.affines import apply_affine
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.special import softmax
import torch
import torch.nn.functional as F
from skimage.measure import label as label_connector
import nibabel as nib
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as edistance
from scipy.ndimage import distance_transform_cdt as chdistance

def remove_zero(f_data, value=0):
    """
    Remove non segmented areas from image
    :param f_data:
    :param value:
    :return:
    """

    xs, ys, zs = np.where(f_data > value)  # find zero values
    tol = 4

    min_max = []
    for x in [xs, ys, zs]:
        minx = min(x) - tol if min(x) - tol > 1 else min(x)
        maxx = max(x) + tol if max(x) + tol < f_data.shape[0] - 1 else max(x)
        min_max.append([minx, maxx])
    f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, min_max

def compute_sdf(segmentation, distance_use='edt', bounded=True):
    """BY B
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    if distance_use=='cdt':
        distance = chdistance
    else:
        distance = edistance

    posmask = segmentation.astype(np.uint8)
    # negmask = ~posmask
    thrs = 4
    from skimage import morphology

    posdis = posmask
    #posdis[posdis > np.max(posdis) / 5] = np.max(posdis) / 5
    #posdis = posdis#/spacing[0]
    #ind = posdis>thrs
    #posdis[ind] = thrs


    #ind_use =  (interior+ exterior + eroded)>0
    #int_ext = (interior+ exterior)>0
    dis = -distance(posmask)

    negdis = distance(1 - posmask)
    dis[(negdis > 0)] = negdis[(negdis > 0)]
    if bounded:
        dis[dis>4] =4
        dis[dis<-4] = -4


    return dis

def binary_fill_holes_lcc_diff(index, threshold=None):
    segl, segl_f = LargestCC(index)
    if len(segl_f) > 2:
        argmax_gmf = np.argsort(segl_f)[-2]
    elif len(segl_f)==2:
        argmax_gmf = 1
    else:
        argmax_gmf = 0
    index_used = (segl != 0) * (segl != argmax_gmf)
    index_used_filled = binary_fill_holes(index_used) > 0
    index_remain = (index_used_filled.astype('int') - index_used.astype('int'))>0
    if threshold is None:
        threshold = (segl_f[argmax_gmf]-1)
    segl_f_remove = np.argwhere(segl_f <= threshold)
    segl_f_keep = np.argwhere(segl_f > threshold)

    if len(segl_f_keep)< len(segl_f_remove):
        shouldbe_kept = 0
        for el in segl_f_keep:
            shouldbe_kept += (segl == el).astype('int')
        shouldbe_removed = ~(shouldbe_kept > 0)
    else:
        shouldbe_removed = 0
        for el in segl_f_remove:
            if el == 0:
                continue
            shouldbe_removed += (segl == el).astype('int')
        if type(shouldbe_removed) == int:
            shouldbe_removed = np.zeros_like(index_used)>0
        else:
            shouldbe_removed = shouldbe_removed>0
    return index_remain, shouldbe_removed

class BiasCorrection(object):
    def __init__(self):
        pass
    def set_info(self, target, reference, weight, biasfield, padding, mask, affine, cov_pq = None,
                 use_original=True):
        self.target = target
        self.reference = reference
        self.weight = weight
        self.biasfield = biasfield
        self.padding = padding
        self.mask = mask
        self.affine = affine
        self.scaler = RobustScaler()
        self.scalerW = StandardScaler()
        self.scalery = StandardScaler()
        self.cov_pq = cov_pq
        self.use_original=use_original

    def _weighted_leas_square(self, imcoord, realdcoord, bias, weights,
                              use_original=True, sample_every= None):

        if sample_every is not None:
            vecB = bias[::sample_every]
            weight = weights[::sample_every]
            realdcoord = realdcoord[::sample_every, :]
        else:
            vecB = bias
            weight = weights

        A = self.biasfield.fit_transform(realdcoord)


        if self.use_original:
            AtW = np.einsum('ji,ik->ji',A.T, weight.reshape(-1,1))
            AtWA = np.matmul(AtW,A)
            AtWy = np.matmul(AtW, vecB.reshape(-1,1))
            invAtWA = np.linalg.inv(AtWA)
            coef = np.matmul(invAtWA, AtWy)
            return coef
        else:
            A = self.scaler.fit_transform(A)

            weight = (weight - weight.min())/np.ptp(weight)
            vecB = self.scalery.fit_transform(vecB.reshape(-1, 1)).squeeze()
            WLS = LinearRegression()

            WLS.fit(A, vecB, sample_weight=1-weight)
            return WLS
    def normalize(self, fi, source):
        a, b = source.min(), source.max()
        dif = b - a
        mindata = fi.min()
        maxdata = fi.max()
        filtered_image = a + (((fi - mindata) * dif) / (maxdata - mindata))
        return filtered_image
    def Apply(self, x, weight=None):
        # apply bias field correction on image
        ind_non_padd = (x != self.padding)* (self.mask==1)
        coord = np.argwhere(ind_non_padd)
        world = apply_affine(self.affine, coord)
        A = self.biasfield.transform(world)
        if self.use_original:
            res = np.matmul(A, self.coef).squeeze()
        else:
            A = self.scaler.transform(A)
            res = self.scalery.inverse_transform(self.coef.predict(A).reshape(-1,1)).squeeze()

        if weight is not None:
            x[ind_non_padd] = x[ind_non_padd] -weight[ind_non_padd]*res
        else:
            x[ind_non_padd] = x[ind_non_padd] -res
        return x

    def Run(self):
        index_selected = (self.target!= self.padding)* (self.mask==1)
        imcoord = np.argwhere(index_selected)
        realdcoord = apply_affine(self.affine, imcoord)
        if self.cov_pq is not None:
            realdcoord = np.concatenate([realdcoord, self.cov_pq[index_selected].reshape(-1,1)], 1)

        bias= (self.target[index_selected] - self.reference[index_selected])#/np.median(self.target[self.target>0])

        weights = self.weight[index_selected]
        # wheighted least square for bias field which polynomial here
        self.coef = self._weighted_leas_square(imcoord, realdcoord, bias=bias, weights=weights, sample_every=None)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))/(np.sqrt(2*np.pi)*sigma) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window_3D(window_size, channel):
    from torch.autograd import Variable
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window









def ssim3D(i1, i2, window, window_size, channel, contrast=True, L=1):
    img1 =torch.from_numpy(i1).unsqueeze(0).unsqueeze(0).to(torch.float)
    img2 = torch.from_numpy(i2).unsqueeze(0).unsqueeze(0).to(torch.float)
    mux = F.conv3d(img1, window, padding=window_size // 2, groups=channel) #Overall Mean Luminance im1
    muy = F.conv3d(img2, window, padding=window_size // 2, groups=channel)#Overall Mean Luminance im2
    mux_sq = mux.pow(2)
    muy_sq = muy.pow(2)
    # Constants for SSIM calculation
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2


    mux_muy = mux * muy

    sigmax_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mux_sq
    sigmax_sq = np.clip(sigmax_sq, 0, sigmax_sq.max())
    sigmay_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - muy_sq
    sigmay_sq = np.clip(sigmay_sq, 0, sigmay_sq.max())
    sigmaxy = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mux_muy
    # structural similarity
    #ssim_map = (sigmaxy + C1) / (sigmax_sq.sqrt() * sigmay_sq.sqrt() + C1)
    #Luminance
    #ssim_map = (2 * mux * muy + C1) / (mux** 2 + muy** 2 + C1)#contrast
    if contrast:
        ssim_map = (2 * sigmax_sq.sqrt() * sigmay_sq.sqrt() + C1) / (sigmax_sq + sigmay_sq + C1)
    else:
        ssim_map = ((2 * mux_muy + C1) * (2 * sigmaxy + C2)) / ((mux_sq + muy_sq + C1) * (sigmax_sq + sigmay_sq + C2))
    ssim_map = ssim_map.squeeze().detach().cpu().numpy()
    return ssim_map

def LargestCC(segmentation, connectivity=3):
    """
    Get largets connected components
    """
    ndim = 3
    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
        ndim = 4
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    # frequency = -np.sort(-frequency)
    return labels, frequency


def update_according_to_neighbours_conv(segmentation, index, label, sign='+',
                                    connectivity=6,kernel_size =3, index_extra=None):
    if kernel_size != 3:
        kernel_size =3
    segmentation_extended = segmentation.reshape(1, 1, *segmentation.shape)
    segmentation_extended = torch.from_numpy(segmentation_extended.astype(np.float32))
    in_out = np.zeros_like(segmentation)
    if sign != '+':
        in_out[index] += 6
    if connectivity==6:
        proxs = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    elif connectivity==26:
        import itertools
        proxs=list(itertools.product([0,1,2], repeat=3))
        proxs.remove((1, 1, 1))

    for i, pr in enumerate(proxs):
        kernel_used = np.zeros(kernel_size).astype(np.float32)
        footprint = np.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)  # mean
        footprint[pr] = 1
        footprint = torch.from_numpy(footprint.reshape(1, 1, *footprint.shape))

        output = F.conv3d(segmentation_extended,
                                  footprint, stride=1,
                                  padding=kernel_size // 2).squeeze().cpu().numpy()
        if index_extra is not None:
            indo = 0
            for l in label:
                indo += (output == l)
            index_sel = ((indo)+index_extra)*index
        else:
            indo = 0
            for l in label:
                indo += (output == l)
            index_sel = (indo)*index
        if sign != '+':
            in_out[index_sel>0] -= 1
        else:
            in_out[index_sel>0] += 1
    return in_out


def neighborhood_conv(output, kerenel_size=3, direction ='x',sqr2dist=False):
    # compute neighborhood pixels
    if sqr2dist: # average of all neighborhood pixels in 3D
        kernel_used = torch.from_numpy(np.zeros(kerenel_size).astype(np.float32))  # mean
        threed_kernel = torch.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)  # mean

        if direction=='xz':
            threed_kernel[0][1][[0,2]] = 1 #xz
            threed_kernel[2][1][[0, 2]] = 1 #xz
        elif direction=='xy':
            threed_kernel[0][0][1] = 1 #xy
            threed_kernel[0][2][1] = 1 #xy

            threed_kernel[2][0][1] = 1 #xy
            threed_kernel[2][2][1] = 1 #xy
        elif direction == 'yz':
            threed_kernel[1][0][[0,2]] =1 # yz
            threed_kernel[1][2][[0,2]] =1 # yz
        elif direction == 'xyz':
            threed_kernel[0][1][[0,2]] = 1 #xz
            threed_kernel[2][1][[0, 2]] = 1 #xz

            threed_kernel[0][0][1] = 1 #xy
            threed_kernel[0][2][1] = 1 #xy

            threed_kernel[2][0][1] = 1 #xy
            threed_kernel[2][2][1] = 1 #xy

            threed_kernel[1][0][[0,2]] =1 # yz
            threed_kernel[1][2][[0,2]] =1 # yz
        else:
            raise exit('Direction should be xy, xz, yz or xyz')

    else:
        kernel_used = torch.from_numpy(np.zeros(kerenel_size).astype(np.float32))  # mean
        threed_kernel = torch.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)
        if direction=='x':

            threed_kernel[0][1][1] = 1.0 # left
            threed_kernel[2][1][1] = 1.0 # right
        elif direction=='y':
            threed_kernel[1][0][1] = 1.0 # left
            threed_kernel[1][2][1] = 1.0 # right
        elif direction == 'z':
            threed_kernel[1][1][0] = 1.0 # left
            threed_kernel[1][1][2] = 1.0  # right
        elif direction == 'xyz':
            threed_kernel[0][1][1] = 1.0 # left
            threed_kernel[2][1][1] = 1.0 # right

            threed_kernel[1][0][1] = 1.0 # left
            threed_kernel[1][2][1] = 1.0 # right

            threed_kernel[1][1][0] = 1.0 # left
            threed_kernel[1][1][2] = 1.0  # right
        else:
            raise exit('Direction should be x, y, y or xyz')
    inp_torch = torch.from_numpy(output.astype(np.float32)).unsqueeze(0).permute([4, 0, 1, 2, 3])
    s = F.conv3d(inp_torch,
                 threed_kernel.reshape(1, 1, *threed_kernel.shape), stride=1,
                 padding=len(kernel_used) // 2)
    s= s.permute([1, 2, 3, 4, 0]).squeeze(0).squeeze(-1).detach().cpu().numpy()
    if s.ndim==3:
        s = s.reshape(*s.shape,1)
    return s

def axis_based_convolution(dif_ven, kernel_size=3, connectivity=6):
    """
    @param larg_dif:
    @param kernel_size:
    @param connectivity:
    @return:
    """
    output = np.repeat(np.expand_dims(np.zeros_like(dif_ven), -1), connectivity, -1)
    dif_ven = dif_ven.reshape(1, 1, *dif_ven.shape)
    dif_ven = torch.from_numpy(dif_ven.astype(np.float32))

    if connectivity==6:
        #'x1,x2, y1,y2,z1,z2'
        proxs=[(0,1,1), (2,1,1), (1,0,1), (1,2,1), (1,1,0), (1,1,2)]

    elif connectivity==26:
        import itertools
        proxs=list(itertools.product([0,1,2], repeat=3))
        proxs.remove((1, 1, 1))
    for i, pr in enumerate(proxs):

        kernel_used = np.zeros(kernel_size).astype(np.float32)
        footprint = np.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)  # mean
        footprint[pr] = 1
        footprint = torch.from_numpy(footprint.reshape(1, 1, *footprint.shape))

        output[...,i] = F.conv3d(dif_ven,
                                    footprint, stride=1,
                                    padding=kernel_size // 2).squeeze().cpu().numpy()
    return output


def adjust_common_structures(prob_r_nonc, threshold=5):
    ind_zero = prob_r_nonc.sum(-1) == 0
    seg_init_nonc = prob_r_nonc.argmax(-1) + 1
    seg_init_nonc[ind_zero] = 0
    neigbs = np.zeros_like(prob_r_nonc)
    concat1 = []
    concat2 = []
    for i in range(prob_r_nonc.shape[-1]):
        ind_ex = seg_init_nonc == (i + 1)
        ind_cc, ind_extra = binary_fill_holes_lcc_diff(ind_ex, threshold=threshold)
        neigbs[..., i] = ind_ex
        concat1.append(ind_extra)
        concat2.append(ind_cc)

    neigbs = neighborhood_conv(neigbs, kerenel_size=3,
                               direction='xyz', sqr2dist=True)

    neigbssf = softmax(neigbs,-1)
    for indice in concat1:
        prob_r_nonc[indice, :] = neigbssf[indice, :]
    for indice in concat2:
        prob_r_nonc[indice, :] = neigbssf[indice, :]
    return prob_r_nonc



def rescale_between_a_b(image, a, b):
    nifti_type = False
    if hasattr(image, 'get_fdata'):
        nifti_type = True
        data_im = image.get_fdata().copy()
    else:
        data_im = image.copy()
    dif = b-a
    mindata= data_im.min()
    maxdata = data_im.max()
    data_im = a + (((data_im - mindata) * dif) / (maxdata - mindata))
    if nifti_type:
        return nib.Nifti1Image(data_im, image.affine, image.header)
    else:
        return data_im