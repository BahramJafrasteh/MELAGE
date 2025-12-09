import os
import csv
import pathlib
import functools
import ants
import numpy as np
import scipy
from skimage import measure

import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, apply_orientation, ornt_transform, inv_ornt_aff
import SimpleITK as sitk
code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features




def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist




def creaePathData_withdir(dir_t1='//MNI_NIFTI', prefix=''):

    imagePath = []
    list_subjects = np.load('LDM_100k/LDM_100k/list_10000.npy.npz')['arr_0']
    list_image = [el for el in os.listdir(dir_t1) if '_1mm_mni.nii' in el]
    for subject in list_image:

        T1 = os.path.join(dir_t1, prefix +subject)
        prob_T1 =  os.path.join(dir_t1,subject.replace('.nii', '_prob1_raw.nii.gz'))
        T2 = os.path.join(dir_t1, subject)
        file_mask_noCSF = subject.replace('_1mm_mni.nii','_1mm_mni_mask.nii' )
        file_mask = subject.replace('_1mm_mni.nii', '_1mm_mni_mask_withCSF.nii.gz')
        mask = os.path.join(dir_t1, file_mask)
        mask_withOutCSF = os.path.join(dir_t1, file_mask_noCSF)
        if mask==T1:
            exit('mask exists')
        if os.path.isfile(T2):
            imagePath.append({'T1': [T1],  'probT1':[prob_T1], 'T2': [T2], 'Mask': [mask], 'Mask_NoCSF': [mask_withOutCSF],
                              'PathTransform':[''], 'PathA_original':[''], 'PathA_prob':[''],'PathA_mask':[''], 'reverseT': [0]})
    return imagePath



def read_atlas_file(atlas_file, printing=False, save_again=False):
    im = nib.load(atlas_file)
    if im.ndim == 4 and im.shape[-1]==1:
        im = nib.funcs.four_to_three(im)[0]
    orig_orient = nib.io_orientation(im.affine)
    code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    source_system = ''.join(list(aff2axcodes(im.affine, code_direction)))
    if source_system != 'RAS':
        print(f'converted to RAS {atlas_file}')
        target_orient = axcodes2ornt('RAS', code_direction)
        transform = ornt_transform(orig_orient, target_orient)
        im = im.as_reoriented(transform)
        if save_again:
            im.to_filename(atlas_file)

    if printing:
        print('{} with range {:.2f}, {:.2f}'.format(atlas_file, im.get_fdata().min(), im.get_fdata().max()))
    #atlas_ims[..., r] = im.get_fdata().copy()
    return im



