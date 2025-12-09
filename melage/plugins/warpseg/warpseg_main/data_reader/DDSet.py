import os.path
from os.path import basename, splitext
import random
random.seed(0)
import torch
import numpy as np
import traceback

import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import os


from .baseData import baseData
from .utils import remove_zero, restore_zero, pad_to_multiple_of_32, createPathData_IXI, creaePathData_LDM_fullhead

import SimpleITK as sitk
from .utils import resample_to_size, read_atlas_file
import nibabel as nib
from .utils import normalize as normalize_image




def softmax_to_logits(softmax_probs, epsilon=1e-6):
    softmax_probs = np.clip(softmax_probs, epsilon, 1 - epsilon)  # Avoid log(0)
    return np.log(softmax_probs) - np.log(1 - softmax_probs)

class DDSet(baseData):
    def options(self, opt, type, novalid):
        """
        options of the data set
        :param opt:
        :return:
        """
        self.opt = opt
        self.state = 'train'
        self.opt = opt

        images_IXI = createPathData_IXI()
        if self.opt.state=='test':
            self.ILpath = images_IXI
        else:
            image_full_head = creaePathData_LDM_fullhead()
            self.ILpath = image_full_head + images_IXI
            self.ILpath = self.ILpath[4000:]


        target_shapes = [None]
        self.y_true = []
        self.probs = []
        self.prob_file = '/atlas/processed/atlas_prob30_raw_fastSurfer.nii.gz'

        prob_values_ = read_atlas_file(self.prob_file, save_again=True)
        prob_values = prob_values_.get_fdata()

        self.target_size = []
        for target_shape in target_shapes:
            #if target_shape==[128, 160, 128]:
            #    print('r')
            y_true, probs, target_size = self.compute_prob_atlas(prob_values, target_shape)
            self.y_true.append(y_true)
            self.probs.append(probs)
            self.target_size.append(target_size)

        self.max_count = self.__len__()+1
        self.new_path = os.path.join(self.opt.chkptDir, 'save_images')
        if not os.path.exists(self.new_path):
            os.makedirs(self.new_path)

    def compute_prob_atlas(self, prob_values, target_shape=None):

        def resize_data(im, mode = 'trilinear'):
            if mode=='nearest':
                im = F.interpolate(torch.from_numpy(im).unsqueeze(0).unsqueeze(0), size=target_size,
                              mode=mode)
            else:
                im = F.interpolate(torch.from_numpy(im).unsqueeze(0).unsqueeze(0), size=target_size,
                             mode=mode, align_corners=False)
            return im

        #(192, 192, 192)
        f_y_true = self.opt.atlas
        #
        if target_shape is None:
            target_size = [224, 256, 192]
            i_y_true = read_atlas_file(f_y_true, save_again=True)
            i_y_true,pad_zero,_ = remove_zero(i_y_true.get_fdata())

            mask = (i_y_true>0)

            mask =resize_data(mask.astype(np.float32), mode='nearest').squeeze(0)>0
            y_true = resize_data(i_y_true).squeeze().detach().cpu().numpy()
            y_true = normalize_image(y_true)
            y_true = torch.from_numpy(y_true).unsqueeze(0)
            y_true[~mask] = 0
            #self.y_true = y_true

            prob_values_new = []
            for ii in range(prob_values.shape[-1]):
                i_y_true = prob_values[...,ii].copy()
                i_y_true, _ ,_ = remove_zero(i_y_true, min_max=pad_zero)
                prob_tm = resize_data(i_y_true).squeeze()
                prob_values_new.append(prob_tm)

            probs = torch.stack(prob_values_new)
            return y_true, probs, target_size
        else:
            i_y_true = read_atlas_file(f_y_true, save_again=True)

            mask = (i_y_true.get_fdata() > 0)
            mask = resample_to_size(
                nib.Nifti1Image(mask, i_y_true.affine, i_y_true.header),
                new_size=target_shape,
                method='nearest').get_fdata()>0

            y_true = resample_to_size(
                nib.Nifti1Image(i_y_true.get_fdata(), i_y_true.affine, i_y_true.header),
                new_size=target_shape,
                method='spline').get_fdata()

            mask,pad_zero ,_ = remove_zero(mask)

            y_true, _ ,_ = remove_zero(y_true, min_max=pad_zero)

            y_true = normalize_image(y_true)
            y_true[~mask] = 0
            y_true, _ = pad_to_multiple_of_32(y_true, target_size=target_shape)
            prob_values_new = []
            for ii in range(prob_values.shape[-1]):
                bg = resample_to_size(
                    nib.Nifti1Image(prob_values[...,ii], i_y_true.affine, i_y_true.header),
                    new_size=target_shape,
                    method='nearest').get_fdata()
                bg, _ ,_ = remove_zero(bg, min_max=pad_zero)
                if ii==0:
                    val_const = 1
                else:
                    val_const = 0
                bg, _ = pad_to_multiple_of_32(bg, target_size=target_shape, val_const=val_const)
                prob_values_new.append(bg)
            probs = np.stack(prob_values_new)
            return y_true, probs, target_shape




    def plot_image_seg(self, imA, imB, outf):
        fig, ax = plt.subplots(2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(imB[:, :, imB.shape[2] // 2])
        ax[1].imshow(imA[:, :, imB.shape[2] // 2])
        plt.savefig(outf)
        plt.close()

    def __getitem__(self, item):
        """
        randomly select one image form the pool
        :param item:
        :return:
        """

        use_registation = True

        ind_image = np.random.randint(len(self.ILpath[item]['T1']))

        pathA = self.ILpath[item]['T1'][ind_image]

        ##'Check is file avialable'
 
        baseNameA = os.path.basename(pathA)
        dirNameA = os.path.dirname(pathA)


        pathMask = self.ILpath[item]['Mask'][ind_image]


        pathMask_noCSF = self.ILpath[item]['Mask_NoCSF'][ind_image]

        pathfile_affine = self.ILpath[item]['PathTransform'][ind_image]
        pathfile_t1_original = self.ILpath[item]['PathA_original'][ind_image]

        pathB = self.ILpath[item]['T2'][ind_image]

        mask_noCSF = read_atlas_file(pathMask_noCSF, save_again=True).get_fdata()>0

        try:

            imA = read_atlas_file(pathA, save_again=True)
            imB = read_atlas_file(pathB, save_again=True)
            mask = read_atlas_file(pathMask, save_again=True).get_fdata()>0
            if mask.sum()<100:
                item = np.random.randint(len(self.ILpath)) % len(self.ILpath)
                return self.__getitem__(item)
            header = imB.header
            affine = imB.affine

            info_back = []
            spacing = imA.header['pixdim'][1:4]

            min_max = [[0, 0], [0, 0], [0, 0]]
            shape_original = imA.shape

            zero_r = False
            if np.random.rand() > 0.5 and self.opt.state!='test':
                zero_r = True

            selcted_size = np.random.choice(len(self.y_true))

            avaial_shape = self.target_size[selcted_size]
            im_shape = imA.shape
            comp_shapes = all([i1 <= i2 for [i1, i2] in zip(*[im_shape, avaial_shape])])
            if use_registation:
                zero_r = True  # for registration
                _mask, _ ,_ = remove_zero(mask)
                im_shape = _mask.shape
                comp_shapes = all([i1 <= i2 for [i1, i2] in zip(*[im_shape, avaial_shape])])

            y_true, probs, target_shape = self.y_true[selcted_size], self.probs[selcted_size], self.target_size[selcted_size]
            image_original = imA.get_fdata()
            if zero_r:
                mask,min_max, shape_original = remove_zero(mask)
                mask_noCSF, _, _ = remove_zero(mask_noCSF, min_max=min_max)
                imA, _ ,_ = remove_zero(imA.get_fdata(), min_max=min_max)
                imB, _ ,_ = remove_zero(imB.get_fdata(), min_max=min_max)
            else:
                imA = imA.get_fdata()
                imB = imB.get_fdata()
            mask = mask == 1

            mask_noCSF = mask_noCSF == 1


            imA = normalize_image(imA, mask)

            imB = normalize_image(imB, mask)

            if np.random.rand() > 0.5 and self.opt.state!='test':
                imA[~mask] = 0
                imB[~mask] = 0


            imA = torch.from_numpy(imA).to(torch.float).unsqueeze(0)

            pad_width = np.array([0,0,0], dtype=np.int32)
            min_max = np.array(min_max, dtype=np.int32)
            shape_original = np.array(shape_original, dtype=np.int32)

            imB = torch.from_numpy(imB).to(torch.float).unsqueeze(0)
            maskT1 = torch.from_numpy(mask).to(torch.int).unsqueeze(0)
            maskT1_noCSF = torch.from_numpy(mask_noCSF).to(torch.int).unsqueeze(0)
            atlas_probs = probs.to(torch.float)
            imout = y_true.to(torch.float)
            image_original = torch.from_numpy(image_original).to(torch.float).unsqueeze(0)
            resized_im = False
            return {'T1': imA,'T1_orig':image_original, 'atlas_probs':atlas_probs,'pathA': pathA, 'atlas': imout,'maskT1': maskT1, 'info_back':info_back,
                    'pixdim':spacing, 'affine':affine, 'pad_width': pad_width, 'min_max': min_max, 'shape_original': shape_original,'resized': resized_im,
                    'maskNoCSF': maskT1_noCSF, 'pathAffine': pathfile_affine, 'pathAOrig':pathfile_t1_original}
        except Exception as e:
            print(pathA)
            print(e)
            #print('Exception occurred:')
            traceback.print_exc()
            #print(min_max)
            #print(shape_original)
            #print(e)
            item = np.random.randint(len(self.ILpath)) % len(self.ILpath)
            return self.__getitem__(item)  

    def __len__(self):
        if self.state.lower() == 'train' or self.state.lower()=='total':
            return len(self.ILpath)
        elif self.state.lower() == 'valid':
            return len(self.ILpath)
        else:
            return len(self.ILpath)

    def name(self):
        return "Tb"






















