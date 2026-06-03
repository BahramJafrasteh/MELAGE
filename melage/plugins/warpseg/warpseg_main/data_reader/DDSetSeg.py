import os
import sys
import random
import json
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
from os.path import basename, splitext
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .baseData import baseData
from .utils import remove_zero
from .utils import normalize as normalize_image
import sys
#sys.modules['tensorflow'] = None
try:
    from monai.transforms import RandRotateD, RandHistogramShiftd, RandGaussianNoised, RandBiasFieldd, ScaleIntensityd, \
    Compose, ToTensor, ToTensorD, RandZoomD, OneOf, EnsureChannelFirstD, EnsureTypeD
except:
    pass
# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

# Set random seed
random.seed(0)

# Define constants
LABEL_MAPPING = {
    'BACKGROUND': 0, 'CORTICAL_GM': 1, 'LAT_VENT_L': 3,
    'CEREBELLUM_CORTEX_L': 5, 'CSF': 15, 'LAT_VENT_R': 19,
    'CEREBELLUM_CORTEX_R': 21,
}


class DDSetSeg(baseData):
    def options(self, opt, type, novalid):
        """
        options of the data set
        :param opt:
        :return:
        """
        self.opt = opt
        self.state = 'train'

        self.target_size = [
            [128, 128, 96], [128, 128, 128], [128, 160, 128],
            [160, 160, 128], [160, 160, 160], [160, 192, 160],
            [192, 192, 160], [192, 224, 192], [192, 256, 192], [224, 256, 192]
        ]

        images_IXI = createPathData_IXI()
        if self.opt.state == 'test':
            self.ILpath =images_IXI #images_OASIS#adni#images_IXI#image_full_head#images_IXI[10:]


        self.train_transforms = Compose([
            # Step A: Add a channel dim ONLY to 'image' and 'mask'.
            # 'prob' is skipped because it's not in the `keys` list.
            EnsureChannelFirstD(keys=["image", "mask"], channel_dim='no_channel'),
            # Step B: Convert all tensors to a float type for calculations.
            EnsureTypeD(keys=["image", "prob", "mask"], dtype=torch.float32),
            OneOf(transforms=[
                RandHistogramShiftd(keys=["image"], prob=0.5),
            ]),
            # Step C: Apply the same affine transform to all keys.
            # This works now because all tensors are 4D (C,D,H,W) and have matching spatial dims.
            RandAffineD(
                keys=["image", "prob", "mask"],
                prob=1.0,  # Using 1.0 for testing
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="zeros"
            )
        ])

        self.max_count = self.__len__() + 1
        self.new_path = os.path.join(self.opt.chkptDir, 'save_images')
        if not os.path.exists(self.new_path):
            os.makedirs(self.new_path)

    def plot_image_seg(self, imA, imB, outf):
        fig, ax = plt.subplots(2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(imB[:, :, imB.shape[2] // 2])
        ax[1].imshow(imA[:, :, imB.shape[2] // 2])
        plt.savefig(outf)
        plt.close()

    def _weight_roi(self, seg, num_classes):
        total_pixels = np.zeros(num_classes)
        class_present = np.zeros(num_classes)

        for c in range(num_classes):
            class_mask = (seg == c)
            pixel_count = class_mask.sum()
            total_pixels[c] = pixel_count
            if pixel_count > 0:
                class_present[c] = 1  # only count if present at all in this volume
        frequencies = total_pixels / (class_present + 1e-6)
        weights = 1.0 / np.log(1.02 + frequencies)  # Smoothed Weighting Formula
        return weights

    def _make_tissue_(self, prob):
        # Core deep gray matter indices
        dgm_indices = [6, 7, 8, 9, 22, 23, 24, 25]
        # Optional (extended) subcortical GM indices — sometimes included in DGM analyses
        extended_dgm_indices = [13, 14, 16, 17, 26, 27, 28, 29]
        dgm_indices += extended_dgm_indices
        prob_tissues = torch.stack(
            [prob[0], prob[[1, 5, 21]].sum(0), prob[[2, 18, 4, 12, 20]].sum(0), prob[[3, 19]].sum(0),
             prob[[10, 11, 15]].sum(0),
             prob[dgm_indices].sum(0)])
        prob_tissues_d = prob_tissues.sum(dim=0, keepdim=True) + 1e-6  # [1, D, H, W]
        prob_tissues = (prob_tissues) / prob_tissues_d
        return prob_tissues

    def __getitem__(self, item):
        """
        randomly select one image form the pool
        :param item:
        :return:
        """
        ind_image = 0

        pathA = self.ILpath[item]['T1'][ind_image]
        pathA_seg = self.ILpath[item]['T1_seg'][ind_image]
        pathA_brain = self.ILpath[item]['T1_brain'][ind_image]
        path_prob_A = self.ILpath[item]['probT1'][ind_image]
        pathMaskA = self.ILpath[item]['Mask'][ind_image]


        zero_r = False
        if np.random.rand() < 0.5:
            zero_r = True

        if not os.path.isfile(path_prob_A) and self.opt.state != 'test':
            next_item = (item + 1) % len(self.ILpath)
            return self.__getitem__(next_item)

        try:
            imA = vxm.py.utils.read_atlas_file(pathA, save_again=True)
            seg = vxm.py.utils.read_atlas_file(pathA_seg, save_again=True).get_fdata()
            imA_brain = vxm.py.utils.read_atlas_file(pathA_brain, save_again=True).get_fdata()
            mask = vxm.py.utils.read_atlas_file(pathMaskA, save_again=True)

            mask = mask.get_fdata() > 0
            # normalize intensities
            header = imA.header
            affine = imA.affine
            spacing = imA.header['pixdim'][1:4]
            min_max = [[0, 0], [0, 0], [0, 0]]
            shape_original = imA.shape
            imA = imA.get_fdata().copy()
            imA = normalize_image(imA, mask)
            imA_brain = normalize_image(imA_brain, mask)
            imA = (np.random.uniform(0.5, 2) * imA_brain * (~mask) + imA)

            prob = vxm.py.utils.read_atlas_file(path_prob_A, save_again=True)
            prob = prob.get_fdata()
            prob[0, seg == 0] = 100

            if len(prob.shape) <= 3:
                prob = None
                seg = None
            if prob is not None:
                num_classes = prob.shape[0]
                weights = self._weight_roi(seg, num_classes)

                if self.opt.state == 'test':
                    zero_r = False
                if zero_r:
                    if np.random.rand() < 0.5 or self.opt.state == 'test':
                        mask, min_max, shape_original = remove_zero(mask)
                      
                        imA, _, _ = remove_zero(imA, min_max=min_max)
                        prob = prob[:, min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1,
                               min_max[2][0]:min_max[2][1] + 1]
                    else:
                        imA = imA * mask
            else:
                weights = None

            avaial_shape = imA.shape

            ind_selcted_size = np.random.choice(len(self.target_size))
            selcted_size = self.target_size[ind_selcted_size]
            
            # The original code had a commented-out line `if self.opt.state=='test':`
            # followed by `selcted_size = self.target_size[-1]`.
            # This implies that for 'test' state, it was intended to always use the last target size.
            # I'm keeping the behavior as it was in the original code, which was to always use the last size,
            # regardless of the `if` condition being commented out.
            selcted_size = self.target_size[-1]

            if prob is not None:
                prob = F.interpolate(torch.from_numpy(prob).unsqueeze(0), size=selcted_size, mode='trilinear',
                                     align_corners=False).squeeze()
            imA = F.interpolate(torch.from_numpy(imA).unsqueeze(0).unsqueeze(0), size=selcted_size, mode='trilinear',
                                align_corners=False).squeeze()
            mask = F.interpolate(torch.from_numpy(mask).to(torch.float).unsqueeze(0).unsqueeze(0), size=selcted_size,
                                 mode='nearest').squeeze()

            sample = {"image": imA, "prob": prob, "mask": mask}
            if self.opt.state != 'test' and np.random.rand() < 0.5:
                sample = self.train_transforms(sample)
                imA = sample['image'].to(torch.float)
                prob = sample['prob'].to(torch.float)
                prob = prob / 100
                prob_tissues = self._make_tissue_(prob)
                mask = sample['mask'].to(torch.float)
                weight_roi = torch.from_numpy(weights).to(torch.float)[:, None, None, None]
            else:
                imA = imA.unsqueeze(0)
                if prob is not None:
                    prob = prob.to(torch.float)
                    prob = prob / 100
                    prob_tissues = self._make_tissue_(prob)
                    weight_roi = torch.from_numpy(weights).to(torch.float)[:, None, None, None]
                else:
                    prob_tissues = None
                mask = mask.to(torch.float).unsqueeze(0)

            if prob_tissues is not None:
                num_classes = prob_tissues.shape[0]
                seg_t = prob_tissues.argmax(0)
                weights_tissue = self._weight_roi(seg_t, num_classes)
                weights_tissue = torch.from_numpy(weights_tissue).to(torch.float)[:, None, None, None]
            else:
                weights_tissue = None
            min_max = np.array(min_max, dtype=np.int32)
            shape_original = np.array(shape_original, dtype=np.int32)
            shape_original = torch.from_numpy(np.array(shape_original)).unsqueeze(0)
            if prob is not None:
                return {'T1': imA.to(torch.float), 'T1_prob': prob.to(torch.float),
                        'T1_prob_edt': weight_roi.to(torch.float), 'pathA': pathA, 'maskT1': mask.to(torch.float),
                        'pixdim': spacing, 'affine': affine, 'min_max': min_max, 'shape_original': shape_original,
                        'interShape': avaial_shape, 'prob_tissue': prob_tissues.to(torch.float),
                        'weight_tissue': weights_tissue.to(torch.float), }
            else:
                return {'T1': imA.to(torch.float), 'pathA': pathA, 'maskT1': mask.to(torch.float),
                        'pixdim': spacing, 'affine': affine, 'min_max': min_max, 'shape_original': shape_original,
                        'interShape': avaial_shape, }

        except Exception as e:
            print(e, pathA)
            item = np.random.randint(len(self.ILpath)) % len(self.ILpath)
            return self.__getitem__(item)

    def __len__(self):
        return len(self.ILpath)

    def name(self):
        return "Tb"






















