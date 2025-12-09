__AUTHOR__ = 'Bahram Jafrasteh'
from .model.mga_net import MGA_NET

import torch
import sys
import os
from model.utils import *
from scipy.ndimage import binary_fill_holes
import torch.nn.functional as F





################## Loading model ################################
def build_model(model_path="MGA_NET.pth", device="cpu"):
    model = MGA_NET(time_embed=True)
    model.to(device)
    if model_path is None:
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "MGA_NET.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'], strict=True)
    return model.eval()



################## Loading data and data standardization ################################
def get_inference(model, imA, device, eco_mri=1, threshold=0.0, high_quality_rec=True):
    time = torch.from_numpy(np.array(eco_mri)).unsqueeze(0).to(torch.float).to(device)

    border_value = imA[0, 0, 0]
    shape_initial = imA.shape
    image_used, pad_zero = remove_zero(imA, border_value)

    image_used = normalize_mri(image_used)/255.0
    shape_zero = image_used.shape

    target_shape = [128, 128, 128]
    #image_used_1 = resample_to_size(nib.Nifti1Image(image_used, affine, header), new_size=target_shape,
    #                                   method='spline').get_fdata()
    image_used_torch = torch.from_numpy(image_used)[None, None, ...].to(torch.float).to(device)
    imB = F.interpolate(image_used_torch, size=target_shape, mode='trilinear', align_corners=False)


    ################## Brain extraction and image reconstruction ################################
    im_low = model.forward(imB, time)
    im_mask_low, im_rec_low = im_low

    if high_quality_rec:

        target_shape = [192, 192, 192] # to create higher quality images
        #image_used_2 = resample_to_size(nib.Nifti1Image(image_used, affine, header), new_size=target_shape,
        #                          method='spline').get_fdata()
        image_used_torch = torch.from_numpy(image_used)[None, None, ...].to(torch.float).to(device)
        im_high_input = F.interpolate(image_used_torch, size=target_shape, mode='trilinear', align_corners=False)

        #imA = torch.from_numpy(image_used_2).to(torch.float).unsqueeze(0).unsqueeze(0)
        #imA = imA.to(device)
        im_high = model.forward(im_high_input, time)

        im_mask_high, im_rec_high = im_high


        im_mask = im_mask_low.detach().cpu()
        im_rec = im_rec_high.detach().cpu()
    else:
        im_mask = im_mask_low.detach().cpu()
        im_rec = im_rec_low.detach().cpu()

    ################## resmaple to the original size ################################
    #im_mask = resample_to_size(nib.Nifti1Image(im_mask, affine, header), new_size=shape_zero,
    #                           method='spline').get_fdata()
    im_mask = F.interpolate(im_mask, size=shape_zero, mode='nearest')
    im_rec = F.interpolate(im_rec, size=shape_zero, mode='trilinear', align_corners=False)
    #im_rec = resample_to_size(nib.Nifti1Image(im_rec, affine, header), new_size=shape_zero,
    #                          method='spline').get_fdata()
    im_mask = im_mask.detach().squeeze().cpu().numpy()
    im_rec = im_rec.detach().squeeze().cpu().numpy()

    im_mask = get_back_data(im_mask, shape_initial, pad_zero, im_mask[0, 0, 0])
    im_rec = get_back_data(im_rec, shape_initial, pad_zero, im_rec[0, 0, 0])


    ind = im_mask >= threshold
    im_mask[ind] = 0
    im_mask[~ind] = 1

    im_mask = binary_fill_holes(im_mask)
    im_mask, labels_freq = LargestCC(im_mask, connectivity=1)
    if len(labels_freq)>2:
        ind_argmax = np.argmax(
            [imA[im_mask == el].sum() for el in range(len(labels_freq)) if el != 0]) + 1
        ind = im_mask != ind_argmax
        im_mask[ind] = 0
        im_mask[~ind] = 1
    im_rec = normalize_mri(im_rec)

    return im_rec, im_mask

def make_ras_image(imA):
    transform, source = convert_to_ras(imA.affine, target='RAS')
    if source != 'RAS':
        imA = imA.as_reoriented(transform)
    return imA
def main():
    file_inp = sys.argv[1]
    eco_mri = int(sys.argv[2])  # -1 for US and 1 for MRI
    threshold = 0.0
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    high_quality_rec = True  # Network has been trained on 128x128x128 size image. However, it is possible to sample 192x192x192 images to get higher quality images
    basen = os.path.basename(file_inp)
    basen = basen[:basen.find('.nii')]

    file_inp_mask = os.path.join(os.path.dirname(file_inp), basen + '_mask.nii.gz')
    file_inp_rec = os.path.join(os.path.dirname(file_inp), basen + '_rec.nii.gz')
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
    # else:
    device = torch.device("cpu")
    model = build_model("MGA_NET.pth", device)
    imA = nib.load(file_inp)

    imA = make_ras_image(imA)

    affine = imA.affine
    header = imA.header
    imB, im_mask = get_inference(model, imA, device)

    imB = nib.Nifti1Image(imB, affine, header).to_filename(file_inp_rec)
    im_mask = nib.Nifti1Image(im_mask, affine, header).to_filename(file_inp_mask)
