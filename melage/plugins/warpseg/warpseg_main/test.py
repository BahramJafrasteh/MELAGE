import os
import sys

# Add project root to Python path to resolve module imports
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#if project_root not in sys.path:
#    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from .voxelmorph.torch.networks import Multi_Segmentor
from .data_reader.utils import normalize as normalize_image, read_atlas_file

# Set environment variables
os.environ['VXM_BACKEND'] = 'pytorch'


from scipy.spatial import cKDTree
import scipy.ndimage as ndi
from .data_reader.utils import LargestCC

def post_process(seg_whole, ind_whole):
    """
    Post-process segmentation to remove small disconnected components 
    and fill holes using nearest neighbor interpolation.
    """
    whole_segs = []
    initial_segs = []
    label = seg_whole.copy()
    
    # 1. Filter connected components
    for c, el in enumerate(ind_whole):
        cur_im = (seg_whole.squeeze() == el)
        initial_segs.append(cur_im)
        
        im_l, cc = LargestCC(cur_im, 1)
        sorted_indices = [idx for idx in np.argsort(cc)[::-1] if idx != 0]
        
        class_mask = np.zeros_like(cur_im, dtype=bool)
        
        if len(sorted_indices) > 0:
            threshold = cc[sorted_indices[0]] * 0.05 
            selected_sorted = [idx for idx in sorted_indices if cc[idx] > threshold]
            
            temp_mask = np.zeros_like(im_l, dtype=int)
            for sl in selected_sorted:
                temp_mask += (im_l == sl).astype(int)
            
            class_mask = temp_mask > 0
            
            class_mask = ndi.binary_fill_holes(class_mask)
        else:
            class_mask = cur_im
            
        whole_segs.append(class_mask)

    initial_segs = np.stack(initial_segs).astype(int)
    whole_segs = np.stack(whole_segs).astype(int)
    
    # 2. Identify changed pixels (removed regions)
    ind_change_whole = (initial_segs - whole_segs) > 0
    
    # 3. Fill removed regions with nearest valid labels
    # Get all valid pixels (those not in the regions being processed or kept regions)
    # Actually, we want to fill using labels from the *final* valid segmentation.
    # But `label` currently has the original values.
    # We should zero out the removed regions in `label` first? 
    # The original code used `label` (original) and found nearest from `valid_indices`.
    # `valid_indices` were defined as pixels in `label` NOT in `ind_whole`.
    # This means it fills using ONLY labels that were NOT part of the post-processed set.
    # This seems to be the intent: fill "holes" (removed bad components) with surrounding "background" or other tissues.
    
    label_indices = np.argwhere(label > 0)
    label_values = label[tuple(label_indices.T)]
    mask_valid = ~np.isin(label_values, ind_whole)
    valid_indices = label_indices[mask_valid]
    valid_values = label_values[mask_valid]
    
    if len(valid_indices) > 0:
        tree = cKDTree(valid_indices)
        
        for ind in range(len(ind_whole)):
            # Pixels that were removed for this class
            indices = np.argwhere(ind_change_whole[ind])
            if len(indices) > 0:
                dist, nearest_idx = tree.query(indices)
                nearest_labels = valid_values[nearest_idx]
                label[tuple(indices.T)] = nearest_labels
                
    return label

def load_model(model, optimizer, device, load_filepath):
    """
    Loads model weights from a file.
    """
    try:
        print(f"Loading model from: {load_filepath}")
        data = torch.load(load_filepath, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {load_filepath}")
        return 0

    # Determine the key for the state dictionary

    pretrained_state = data["model"]



    # Using strict=False is flexible for mismatched keys
    incompatible_keys = model.load_state_dict(pretrained_state, strict=False)
    if incompatible_keys.missing_keys:
        print(f"Warning: Missing keys in state_dict: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")
    #except Exception as e:
    #    print(f"Error while loading state dict: {e}")
        #return 0
    
    current_step = 0
    try:
        if optimizer is not None:
            optimizer.load_state_dict(data['opt'])
        current_step = data['step']
    except KeyError:
        print("Optimizer state or step not found in checkpoint. Starting from step 0.")
    return 1

def save_segmentation(segmentation, affine, base_name, out_folder, suffix='_seg_ours.nii.gz'):
    """
    Saves the segmentation result as a NIfTI file.
    """
    if '.nii.gz' in base_name:
        new_filename = base_name.replace('.nii.gz', suffix)
    elif '.nii' in base_name:
        new_filename = base_name.replace('.nii', suffix)
    else:
        new_filename = base_name + suffix
        
    out_file_path = os.path.join(out_folder, new_filename)
    nifti_image = nib.Nifti1Image(segmentation, affine, header=None)
    nifti_image.to_filename(out_file_path)
    print(f"Saved: {out_file_path}")

def get_inference(model, input_data, device, inshape=[224, 256, 192],
                  post_processing=True):
    input_data = normalize_image(input_data)

    # Prepare tensor
    input_tensor = torch.from_numpy(input_data).float().to(device)
    # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    # Assuming input is 3D volume
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.ndim == 4:  # Maybe already has channel?
        input_tensor = input_tensor.unsqueeze(0)

    source_shape = input_tensor.shape[2:]

    # Resize to model input shape
    input_tensor_resized = F.interpolate(input_tensor, size=inshape, mode='trilinear', align_corners=False)

    print(f"Running inference...")
    with torch.inference_mode():
        # Inference
        pred_logits, pred_logits_tissue = model(input_tensor_resized)

        # Resize back to original shape
        pred_logits = F.interpolate(pred_logits, size=source_shape, mode='trilinear', align_corners=False)
        pred_logits_tissue = F.interpolate(pred_logits_tissue, size=source_shape, mode='trilinear', align_corners=False)

        # Get probabilities and segmentations
        prob_value = F.softmax(pred_logits, dim=1)
        prob_value_tissue = F.softmax(pred_logits_tissue, dim=1)

        segmentation = prob_value.argmax(1).squeeze().numpy().astype(np.int16)
        if post_processing:
            segmentation = post_process(segmentation, ind_whole=[3, 19])
        segmentation_tissue = prob_value_tissue.argmax(1).squeeze().numpy().astype(np.int16)
        if post_processing:
            segmentation_tissue = post_process(segmentation_tissue, ind_whole=[3])

        # Convert to numpy
        #out_segmentation = segmentation.detach().squeeze().cpu().numpy().astype(np.int16)
        #out_segmentation_tissue = segmentation_tissue.detach().squeeze().cpu().numpy().astype(np.int16)
    return segmentation_tissue, segmentation


def build_model(model_path, device):
    inshape = [224, 256, 192]
    model = Multi_Segmentor(tissue_roi=6, nrois=30).to(device)

    # Load model
    if model_path is not None:
        step = load_model(model, None, device, load_filepath=model_path)
        if step == 0:
             raise RuntimeError(f"Failed to load model weights from {model_path}")
    model.to(device)
    model.eval()
    return model

def get_options():
    parser = argparse.ArgumentParser()

    # Input/Output parameters
    parser.add_argument('--input-image', required=True, help='Path to the input image for inference')
    parser.add_argument('--output-dir', default='test_results', help='Directory to save results')
    parser.add_argument('--model-path', required=True, help='Path to the trained model checkpoint')

    # Model parameters (must match training config)
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # Other parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--modelName', type=str, default='t1_fuzzy',
                        help='Automatic')  # Kept for compatibility if needed

    opt = parser.parse_args()
    return opt

def main():
    import voxelmorph as vxm
    opt = get_options()
    
    # Device configuration
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Model configuration
    inshape = [224, 256, 192]
    model = build_model(opt.model_path, device)

    # Prepare output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Load and process input image
    if not os.path.exists(opt.input_image):
        print(f"Error: Input image not found at {opt.input_image}")
        return

    try:
        # Load image
        nifti_img = read_atlas_file(opt.input_image)
        input_data = nifti_img.get_fdata()
        affine = nifti_img.affine
        
        # Normalize
        out_segmentation_tissue, out_segmentation = get_inference(model, input_data, device, inshape)
            
        # Save results
        base_name = os.path.basename(opt.input_image)
        save_segmentation(out_segmentation, affine, base_name, opt.output_dir, suffix='_seg_ours.nii.gz')
        save_segmentation(out_segmentation_tissue, affine, base_name, opt.output_dir, suffix='_tissue_seg_ours.nii.gz')
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
