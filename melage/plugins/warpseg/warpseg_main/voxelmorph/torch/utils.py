import torch
import torch.nn.functional as F
import torch.nn as nn
import ants
import nibabel as nib

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

def intensity_res(imT1, prob):
    # Avoid division by zero
    eps = 1e-5

    # Ensure input tensors are on the same device (e.g., CUDA or CPU)
    imT1 = imT1.float()  # Ensure imT1 is a float tensor
    prob = prob.float()  # Ensure prob is a float tensor

    # Broadcast imT1 to match the shape of prob (B, K, D, H, W)
    if imT1.shape[1] == 1:
        imT1_broadcast = imT1.expand(-1, prob.shape[1], -1, -1, -1)  # Broadcast imT1 to (B, K, D, H, W)
    else:
        imT1_broadcast = imT1  # (B, K, D, H, W), no need to broadcast if shapes already match

    # Mean: mu_k shape (B, K)
    sum_prob = torch.sum(prob, dim=(2, 3, 4))  # Sum over the spatial dimensions
    mu_k = torch.sum(prob * imT1_broadcast, dim=(2, 3, 4)) / (sum_prob + eps)

    # Variance: var_k shape (B, K)
    mu_k_broadcast = mu_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1, 1) for broadcasting
    var_k = torch.sum(prob * (imT1_broadcast - mu_k_broadcast) ** 2, dim=(2, 3, 4)) / (sum_prob + eps)

    # Standard deviation: std_k shape (B, K)
    std_k = torch.sqrt(var_k + eps)

    # Expand for broadcasting
    mu = mu_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (B, K, 1, 1, 1)
    std = std_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape (B, K, 1, 1, 1)

    # Log-likelihood under Gaussian model
    log_likelihood = -0.5 * ((imT1_broadcast - mu) ** 2) / (std ** 2 + eps) \
                     - torch.log(std + eps) \
                     - 0.5 * np.log(2 * np.pi)

    # Softmax across regions (K) to get confidence weights
    logits = log_likelihood
    logits_max = torch.max(logits, dim=1, keepdim=True).values  # Numerical stability (subtract max)
    exp_logits = torch.exp(logits - logits_max)  # for numerical stability
    confidence_weights = exp_logits / (torch.sum(exp_logits, dim=1, keepdim=True) + eps)  # shape (B, K, D, H, W)

    return confidence_weights

def create_image_from_prob(imT1, prob_value):
    topk_probs, topk_labels = torch.topk(prob_value, k=1, dim=1)

    centers = (prob_value * imT1).sum(dim=[2, 3, 4]) / (prob_value.sum(dim=[2, 3, 4]) + 1e-5)  # shape [B, C]
    # Reshape centers for broadcasting: [B, C, 1, 1, 1]
    centers_expanded = centers.view(centers.shape[0], centers.shape[1], 1, 1, 1)
    # Expand to match spatial dims of topk_labels: [B, C, D, H, W]
    centers_expanded = centers_expanded.expand(-1, -1, *topk_labels.shape[2:])
    # Make sure topk_labels is shaped for indexing: [B, 1, D, H, W] (for class/channel selection)
    # It should be long/int type
    topk_labels = topk_labels.long()  # Add channel dim for gather
    # Gather values: [B, 1, D, H, W]
    new_image = torch.gather(centers_expanded, dim=1, index=topk_labels)

    return new_image

def write_to_file(prob_atlas_size,mask_original, mask_new, prob_new,  pathOrigin):
    imA = read_atlas_file(pathOrigin, rewrite_to_file=True)

    mask_original = mask_original.squeeze().detach().cpu().numpy()
    mask_original = nib.Nifti1Image(mask_original.astype(np.int16), affine=imA.affine, header=imA.header)

    mask_original.to_filename(mask_new)

    prob_atlas_size = prob_atlas_size.squeeze().detach().cpu().numpy()
    prob_atlas_size = nib.Nifti1Image(prob_atlas_size, affine=imA.affine, header=imA.header)

    prob_atlas_size.to_filename(prob_new)
    print(f'OutProb_original: {prob_new}')



class Histogram_Matching(nn.Module):
    def __init__(self, differentiable=True, bins=32, sigma=None):
        super(Histogram_Matching, self).__init__()
        self.differentiable = differentiable
        self.bins = bins
        self.sigma = sigma if sigma is not None else 5.0

    def forward(self, dst, ref):
        B, C, D, H, W = dst.shape
        assert dst.device == ref.device

        hist_dst = self.cal_hist(dst)
        hist_ref = self.cal_hist(ref)

        tables = self.cal_trans_batch(hist_dst, hist_ref)

        dst_flat = dst.view(B*C, -1) * (self.bins - 1)  # scale intensities
        dst_flat = dst_flat.unsqueeze(2)  # (B*C, N, 1)

        indices = torch.linspace(0, self.bins-1, self.bins, device=dst.device).view(1, 1, -1)  # (1,1,bins)

        weights = torch.softmax(-((dst_flat - indices)**2) / (2*(self.sigma**2)), dim=-1)  # (B*C, N, bins)

        matched = torch.bmm(weights, tables.unsqueeze(-1)).squeeze(-1)  # (B*C, N)

        matched = matched.view(B, C, D, H, W) / (self.bins - 1)  # back to [0,1]

        return matched

    def cal_hist(self, img):
        B, C, D, H, W = img.shape
        if self.differentiable:
            hists = self.soft_histc_batch(img * (self.bins - 1))
        else:
            hists = torch.stack([
                torch.histc(img[b, c] * (self.bins - 1), bins=self.bins, min=0, max=self.bins-1)
                for b in range(B) for c in range(C)
            ])
        hists = F.normalize(hists.float(), p=1, dim=1)
        cdfs = torch.cumsum(hists, dim=1)
        return cdfs

    def soft_histc_batch(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B*C, -1)

        delta = 1.0
        centers = torch.linspace(0, self.bins-1, self.bins, device=x.device).float()
        centers = centers[None, :, None]  # (1, bins, 1)

        x = x.unsqueeze(1)  # (B*C, 1, N)

        soft_bins = torch.sigmoid((x - centers + delta/2) * self.sigma) - torch.sigmoid((x - centers - delta/2) * self.sigma)
        soft_hist = soft_bins.sum(dim=2)  # (B*C, bins)

        return soft_hist

    def cal_trans_batch(self, cdf_src, cdf_ref):
        Bc, bins = cdf_src.shape
        cdf_src = cdf_src.unsqueeze(2)  # (B*C, bins, 1)
        cdf_ref = cdf_ref.unsqueeze(1)  # (B*C, 1, bins)

        dist = torch.abs(cdf_src - cdf_ref)  # (B*C, bins, bins)

        mapping = dist.argmin(dim=2)  # (B*C, bins)

        return mapping.float()

def erode_mask(mask, radius=1):
    kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1, 2*radius+1), device=mask.device)
    padded = F.pad(mask.float(), [radius]*6, mode='replicate')
    eroded = F.conv3d(padded, kernel) == kernel.numel()
    return eroded.float()

def adjust_probs_with_uncertainty(probs, uncertainty, min_temp=1.0, max_temp=5.0):
    # Normalize uncertainty to [0, 1]
    norm_unc = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-5)

    # Convert normalized uncertainty to temperature (higher temp = more smoothing)
    temp = min_temp + (max_temp - min_temp) * norm_unc  # shape: (B, 1, D, H, W)

    # To avoid instability, apply temp scaling in log-space
    log_probs = torch.log(probs + 1e-8)
    log_probs_scaled = log_probs / temp  # broadcast over channels

    probs_adj = F.softmax(log_probs_scaled, dim=1)
    return probs_adj

def get_back_to_size_torch(img: torch.Tensor, pad_width):
    (pad_x, pad_y, pad_z) = pad_width
    pad_x1, pad_x2 = pad_x
    pad_y1, pad_y2 = pad_y
    pad_z1, pad_z2 = pad_z

    unpadded_img = img[
        ...,
        pad_x1 : -pad_x2 if pad_x2 > 0 else None,
        pad_y1 : -pad_y2 if pad_y2 > 0 else None,
        pad_z1 : -pad_z2 if pad_z2 > 0 else None,
    ]
    return unpadded_img


def calculate_jacobian_determinant_3d(flow):
    """Calculates the Jacobian determinant of a 3D deformation field."""
    B, _, D, H, W = flow.shape
    device = flow.device
    grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(D, device=device), torch.arange(H, device=device),
                                            torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y, grid_z), dim=0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    transform_map = grid + flow

    diff_x = transform_map[..., 1:] - transform_map[..., :-1]
    grad_x = torch.cat([diff_x, diff_x[..., -1:]], dim=-1)
    diff_y = transform_map[..., 1:, :] - transform_map[..., :-1, :]
    grad_y = torch.cat([diff_y, diff_y[..., -1:, :]], dim=-2)
    diff_z = transform_map[:, :, 1:, :, :] - transform_map[:, :, :-1, :, :]
    grad_z = torch.cat([diff_z, diff_z[:, :, -1:, :, :]], dim=-3)

    J_xx, J_yx, J_zx = grad_x[:, 0, ...], grad_x[:, 1, ...], grad_x[:, 2, ...]
    J_xy, J_yy, J_zy = grad_y[:, 0, ...], grad_y[:, 1, ...], grad_y[:, 2, ...]
    J_xz, J_yz, J_zz = grad_z[:, 0, ...], grad_z[:, 1, ...], grad_z[:, 2, ...]

    determinant = (J_xx * (J_yy * J_zz - J_zy * J_yz) - J_xy * (J_yx * J_zz - J_zx * J_yz) + J_xz * (
                J_yx * J_zy - J_zx * J_yy))
    return determinant.unsqueeze(1)


def spatial_transformer_field(image, deformation_field, mode='bilinear'):
    device = image.device
    D, H, W = image.shape[2:]
    d_coords, h_coords, w_coords = torch.meshgrid(torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    identity_grid = torch.stack([w_coords, h_coords, d_coords], dim=-1).unsqueeze(0).to(device)
    scaling = torch.tensor([2 / (W - 1), 2 / (H - 1), 2 / (D - 1)], device=device)
    deformation_field_normalized = deformation_field.permute(0, 2, 3, 4, 1) * scaling
    sampling_grid = identity_grid + deformation_field_normalized
    if mode=='bilinear':
        warped_image = F.grid_sample(image, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    elif mode=='nearest':
        warped_image = warped_image = F.grid_sample(image, sampling_grid, mode='nearest', padding_mode='border', align_corners=True)
    else:
        return None
    return warped_image

def augment_based_on_deform_field(source,  atlas_probs):
    # Define which ROIs you want to smooth



    IMAGE_SHAPE = source.shape[2:]
    DEFORMATION_COARSE_GRID = (4, 2, 4)
    DEFORMATION_MAGNITUDE = 8.0  # Max displacement in pixels
    deformation_field = generate_smooth_deformation_field(
        shape=IMAGE_SHAPE,
        coarse_grid_shape=DEFORMATION_COARSE_GRID,
        magnitude=DEFORMATION_MAGNITUDE
    )
    print("Calculating Jacobian determinant...")
    jacobian_map = calculate_jacobian_determinant_3d(deformation_field)
    y_source = spatial_transformer_field(source, deformation_field)
    prob_atlas = spatial_transformer_field(atlas_probs, deformation_field, mode='nearest')  # *mask
    # c. Modulate the intensity of the warped image using the Jacobian
    print("Modulating intensity based on deformation...")
    INTENSITY_EFFECT_STRENGTH = 0.8  # How much compression/expansion changes brightness
    INTENSITY_NOISE_LEVEL = 0.01  # How much random noise to add
    modulated_image = modulate_intensity_with_jacobian(
        y_source, prob_atlas, jacobian_map,
        effect_strength=INTENSITY_EFFECT_STRENGTH,
        noise_level=INTENSITY_NOISE_LEVEL
    )
    return modulated_image, prob_atlas

# --- NEW FUNCTION ---
def modulate_intensity_with_jacobian(base_image, atlas, jacobian_map, effect_strength=0.5, noise_level=0.1):
    """
    Modulates image intensity within ROIs based on local Jacobian values.

    Args:
        base_image (torch.Tensor): The image to modify (1, 1, D, H, W).
        atlas (torch.Tensor): Atlas with integer labels for ROIs (1, 1, D, H, W).
        jacobian_map (torch.Tensor): The Jacobian determinant map (1, 1, D, H, W).
        effect_strength (float): How strongly the Jacobian affects intensity.
                                 Positive: compression -> brighter.
                                 Negative: compression -> darker.
        noise_level (float): The standard deviation of Gaussian noise to add.

    Returns:
        torch.Tensor: The image with modulated intensities.
    """
    #modulated_image = base_image.clone()

    pixels_in_roi = base_image
    jacobian_in_roi = jacobian_map

    # 1. Calculate the deterministic intensity modulation factor
    # (J-1) centers the effect: J=1 -> no change. J<1 (compression) -> negative.

    modulation_factor = 1.0 - (jacobian_in_roi - (jacobian_in_roi.mean()+2*jacobian_in_roi.std())) * effect_strength

    # 2. Create a random noise component
    random_noise = torch.randn_like(pixels_in_roi) * noise_level

    # 3. Apply the changes
    modified_pixels = pixels_in_roi * modulation_factor + random_noise

    # Place the modified pixels back into the image
    modulated_image = modified_pixels

    """
    

    # Process each ROI label found in the atlas
    for label in torch.unique(atlas)[1:]:  # Skip background (label 0)
        mask = (atlas == label)

        # Get the pixels and corresponding Jacobian values for this ROI
        pixels_in_roi = base_image[mask]
        jacobian_in_roi = jacobian_map[mask]

        # 1. Calculate the deterministic intensity modulation factor
        # (J-1) centers the effect: J=1 -> no change. J<1 (compression) -> negative.
        modulation_factor = 1.0 - (jacobian_in_roi - 1.0) * effect_strength

        # 2. Create a random noise component
        random_noise = torch.randn_like(pixels_in_roi) * noise_level

        # 3. Apply the changes
        modified_pixels = pixels_in_roi * modulation_factor + random_noise

        # Place the modified pixels back into the image
        modulated_image[mask] = modified_pixels
    """
    return modulated_image
def generate_smooth_deformation_field(shape, coarse_grid_shape, magnitude):

    """
    Generates a smooth, random 3D deformation field.

    This works by creating a very low-resolution random vector field and then
    upsampling and smoothing it into a high-resolution, plausible deformation.

    Args:
        shape (tuple): The final shape of the field (D, H, W).
        coarse_grid_shape (tuple): The low-resolution grid size (e.g., (4, 4, 4)).
        magnitude (float): A scaling factor for the displacement strength.

    Returns:
        torch.Tensor: The smooth deformation field of shape (1, 3, D, H, W).
    """
    # 1. Create the coarse, random vector field
    coarse_field = np.random.randn(*coarse_grid_shape, 3) * magnitude

    # 2. Upsample the coarse field to the final shape
    zoom_factors = [s / cs for s, cs in zip(shape, coarse_grid_shape)]

    # Upsample each displacement channel (x, y, z)
    upsampled_field = np.zeros((*shape, 3))
    for i in range(3):
        upsampled_field[..., i] = zoom(coarse_field[..., i], zoom=zoom_factors, order=3)  # Cubic spline interpolation

    # 3. Smooth the upsampled field to make it physically plausible
    # A large sigma creates very smooth, large-scale deformations.
    smoothed_field = np.zeros_like(upsampled_field)
    for i in range(3):
        smoothed_field[..., i] = gaussian_filter(upsampled_field[..., i], sigma=10)

    # 4. Convert to a PyTorch tensor and format for the spatial transformer
    # The channels need to be in the second dimension (B, C, D, H, W)
    # The standard for grid_sample is (x, y, z), which corresponds to (W, H, D)
    #flow_tensor = torch.from_numpy(smoothed_field).permute(3, 0, 1, 2).unsqueeze(0).float()

    # PyTorch flow fields are ordered (x,y,z), corresponding to W,H,D dimensions
    # Our numpy array was D,H,W,C so we permute C,D,H,W
    # Let's re-order the numpy array for clarity before converting to tensor
    smoothed_field_xyz = smoothed_field[..., [2, 1, 0]]  # Reorder to x,y,z
    flow_tensor = torch.from_numpy(smoothed_field_xyz).permute(3, 0, 1, 2).unsqueeze(0).float()

    return flow_tensor


def restore_zero_torch(cropped_data: torch.Tensor, original_shape, min_max, value=0):
    """
    Restore the cropped data back to the original shape with padding (default = 0).

    :param cropped_data: The cropped tensor from remove_zero
    :param original_shape: The shape of the original full image
    :param min_max: The min/max indices used for cropping
    :param value: The padding value to fill
    :return: Restored full-size tensor
    """
    if min_max.abs().sum()==0:
        return cropped_data
    B, C, _, _, _ = cropped_data.shape
    W,H, D = original_shape
    restored = torch.full((B,C,W,H,D), value, dtype=cropped_data.dtype, device=cropped_data.device)

    restored[:,:,
    min_max[0][0]:min_max[0][1] + 1,
    min_max[1][0]:min_max[1][1] + 1,
    min_max[2][0]:min_max[2][1] + 1
    ] = cropped_data

    return restored

def change_to_original(im_changed, data, batch_num=0):
    im1 = get_back_to_size_torch(im_changed, data['pad_width'][batch_num])
    im1 = restore_zero_torch(im1, data['shape_original'][batch_num], data['min_max'][batch_num])
    return im1

def differentiable_boundary(mask, kernel_size=3):
    """Returns soft boundary: high when neighbor differs from center."""
    # Laplacian-like filter via average pooling
    pad = kernel_size // 2
    avg_mask = F.avg_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.abs(avg_mask - mask)


def laplacian_3d(image):
    # 3D Laplacian kernel (center - 6 neighbors)
    kernel = torch.tensor([[[[0, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]],

                            [[0, -1, 0],
                             [-1, 6, -1],
                             [0, -1, 0]],

                            [[0, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]]]], dtype=image.dtype, device=image.device)

    kernel = kernel.unsqueeze(0)  # shape: [1, 1, 3, 3, 3]

    # Apply convolution
    lap = F.conv3d(image, kernel, padding=1)
    return lap
def soft_erode(img, iterations=3):
    for _ in range(iterations):
        img = -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)
    return img

def soft_skeletonize(img, iterations=3):
    skel = F.relu(img - soft_erode(img, iterations))
    return skel
def gradient_3d(img):
    dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    dy = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dx = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]
    return dx, dy, dz

def bending_energy_loss(disp):
    """
    disp: tensor of shape (B, 3, D, H, W), displacement field
    """
    dx = disp[:, :, 2:, 1:-1, 1:-1] - 2 * disp[:, :, 1:-1, 1:-1, 1:-1] + disp[:, :, :-2, 1:-1, 1:-1]
    dy = disp[:, :, 1:-1, 2:, 1:-1] - 2 * disp[:, :, 1:-1, 1:-1, 1:-1] + disp[:, :, 1:-1, :-2, 1:-1]
    dz = disp[:, :, 1:-1, 1:-1, 2:] - 2 * disp[:, :, 1:-1, 1:-1, 1:-1] + disp[:, :, 1:-1, 1:-1, :-2]
    return (dx.pow(2) + dy.pow(2) + dz.pow(2)).mean()

def gm_highest_at_wm_border(P_WM, P_GM, P_BG, P_GM_previous, eps=1e-6):
    """
    Computes a soft, differentiable metric for:
    where WM touches CSF or BG, is GM the highest?
    """


    # Extract individual tissue probs
    wm = P_WM
    gm = P_GM
    bg = P_BG

    # Define WM border using soft boundary detection
    kernel = torch.zeros((1, 1, 3, 3, 3), device=P_BG.device)
    offsets = [(0, 1, 1), (1, 0, 1), (1, 1, 0),
               (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    for i, j, k in offsets:
        kernel[0, 0, i, j, k] = 1.0
    neighbour_wm = F.conv3d((wm > 0.5).float(), kernel, padding=1)/6.  # (B, 1, H, W, D)
    wm_boundary = (neighbour_wm-wm).abs()
    # Compute contact with CSF or BG: high where neighboring probs are nonzero
    csf_or_bg = ( (bg[0]==0)).float()
    touching = (wm_boundary * csf_or_bg)>0  # soft mask for "touching" regions
    gm[touching] = P_GM_previous[touching]

    return gm  # in [0, 1], higher = more GM dominance at WM borders