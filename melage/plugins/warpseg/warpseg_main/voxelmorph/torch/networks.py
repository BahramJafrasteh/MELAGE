import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import nibabel as nib
from . import layers
from .modelio import LoadableModel, store_config_args
from .multi_stage_net import MultiStageNet
from .utils import (change_to_original, gm_highest_at_wm_border, write_to_file, restore_zero_torch)
from .Unet import Unet
import os
class Multi_Segmentor(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 tissue_roi=6,
                 nrois=30):
        super().__init__()
        # image segmentation
        self.seg_model = MultiStageNet(channels=nrois, tissue_channels=tissue_roi)

    def forward(self, source):
        _, _, D, H, W = source.shape
        pred_logits, pred_logits_seg = self.seg_model(source)
        return pred_logits, pred_logits_seg




class FVxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self, opt,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 nrois=30):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        self.opt = opt

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir
        self.int_downsize = int_downsize
        self.int_steps= int_steps
        self.create_integrate(inshape, )

    def create_integrate(self, inshape, device=None):

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / self.int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, self.int_steps) if self.int_steps > 0 else None


        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

        self.transfomer_prob = layers.SpatialTransformer(inshape, mode='nearest')

        if device is not None:
            self.integrate.to(device)
            self.transformer.to(device)
            self.transfomer_prob.to(device)

    def forward(self, source, target, mask, atlas_probs=None, registration=False, mode='bilinear',
                data_info=None, mask_noCSF=None, use_registration=True, probs_input=None):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        inshape = source.shape[2:]
        device = source.device
        self.create_integrate(inshape, device)
        _,_, D, H, W = target.shape
        shape_orig = [D, H, W]
        _,_, tD, tH, tW = source.shape
        interp = False
        if D != tD or H != tH or W != tW:
            interp = True
            target = F.interpolate(target, size=source.shape[2:], mode='trilinear', align_corners=False)
            mask = F.interpolate(mask.float(), size=source.shape[2:], mode='nearest')
            if mask_noCSF is not None:
                mask_noCSF = F.interpolate(mask_noCSF.float(), size=source.shape[2:], mode='nearest')
            else:
                mask_noCSF = mask

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None



        if mode == 'bilinear':
            # warp image with flow field
            y_source = self.transformer(source, pos_flow)
            y_target = self.transformer(target, neg_flow) if self.bidir else None
        elif mode == 'nearest':
            y_source = self.transfomer_prob(source, pos_flow)
            y_target = self.transfomer_prob(target, neg_flow) if self.bidir else None

        prob_atlas = self.transformer(atlas_probs, pos_flow)

        if probs_input is not None:
            prob_warped_input = self.transfomer_prob(probs_input, neg_flow)  # *mask
        else:
            prob_warped_input = None


        if not use_registration:
            no_prob = prob_atlas.sum(1) == 0
            prob_atlas[:, 0][no_prob] = 1 # background
            gm_previous = prob_atlas[:, 1].clone() #GM
            prob_atlas[:, 1] *= mask_noCSF[:, 0]
            prob_atlas[:, 2] *= mask_noCSF[:, 0] #WM
            prob_atlas[:, 18] *= mask_noCSF[:, 0] #WM
            gm_updated = gm_highest_at_wm_border(P_WM=prob_atlas[:, 2] + prob_atlas[:, 18], P_GM=prob_atlas[:, 1],
                                                 P_BG=mask_noCSF, P_GM_previous=gm_previous)
            prob_atlas[:, 1] = gm_updated

            if 2>1:

                pathA = data_info['pathA'][0]
                baseNameA = os.path.basename(pathA)
                dirNameA = os.path.dirname(pathA)

                if '.nii.gz' in baseNameA:
                    prob_newA = os.path.join(dirNameA, baseNameA.replace('.nii.gz', '_prob1_raw.nii.gz'))
                    pos_flow_file = os.path.join(dirNameA, baseNameA.replace('.nii.gz', '_df.nii.gz'))

                elif '.nii' in baseNameA:
                    prob_newA = os.path.join(dirNameA, baseNameA.replace('.nii', '_prob1_raw.nii.gz'))
                    pos_flow_file = os.path.join(dirNameA, baseNameA.replace('.nii', '_df.nii.gz'))
                if interp:

                    prob_atlas = F.interpolate(prob_atlas, size=shape_orig, mode='trilinear', align_corners=False)
                    mask = F.interpolate(mask, size=shape_orig, mode='trilinear', align_corners=False)
                    mask_noCSF = F.interpolate(mask_noCSF, size=shape_orig, mode='trilinear', align_corners=False)
                    pos_flow = F.interpolate(pos_flow, size=shape_orig, mode='trilinear', align_corners=False)

                    batch_num = 0
                    par1, par2 = data_info['shape_original'][batch_num], data_info['min_max'][batch_num]
                    prob_atlas_size = restore_zero_torch(prob_atlas, par1, par2)
                    mask_original = restore_zero_torch(mask, par1, par2)
                    mask_noCSF = restore_zero_torch(mask_noCSF, par1, par2)
                    pos_flow = restore_zero_torch(pos_flow, par1, par2)

                else:
                    # previous configuration
                    prob_atlas_size = change_to_original(prob_atlas, data_info, batch_num=0)
                    mask_original = change_to_original(mask, data_info, batch_num=0)

                if not os.path.isfile(prob_newA):
                    imA = nib.load(pathA)

                    prob_atlas= prob_atlas_size.squeeze().detach().cpu().numpy()
                    mask_noCSF = mask_noCSF.squeeze().detach().cpu().numpy()
                    pos_flow = pos_flow.squeeze().detach().cpu().numpy()

                    probA= nib.Nifti1Image(prob_atlas, affine=imA.affine, header=imA.header )
                    probA.to_filename(prob_newA)

                    probA= nib.Nifti1Image(pos_flow, affine=imA.affine, header=imA.header )
                    probA.to_filename(pos_flow_file)
                    print(f'OutProb: {prob_newA}')
                if not data_info['need_reverse'][0].item() == 1:
                    return

                if os.path.isfile(prob_newA):
                    return

                pathOrigin = data_info['pathAOrig'][0]
                baseName = os.path.basename(pathOrigin)
                dirName = os.path.dirname(pathOrigin)
                if '.nii.gz' in baseName:
                    prob_new = os.path.join(dirName, baseName.replace('.nii.gz', '_original_prob.nii.gz'))
                    mask_new = os.path.join(dirName, baseName.replace('.nii.gz', '_original_mask.nii.gz'))
                elif '.nii' in baseName:
                    prob_new = os.path.join(dirName, baseName.replace('.nii', '_original_prob.nii.gz'))
                    mask_new = os.path.join(dirName, baseName.replace('.nii', '_original_mask.nii.gz'))
                if os.path.isfile(prob_new) and os.path.isfile(mask_new):
                    return


                target_1 = data_info['T1_orig'].to(source.device)
                #target_1, mask_original, prob_atlas_size = reverse_transform_image_labels(target_1, mask_original,
                                                                                          #prob_atlas_size, data_info)

                write_to_file(prob_atlas_size,mask_original, mask_new, prob_new,  pathOrigin)
                return target_1, mask_original, prob_atlas_size
            else:
                return
        else:
            pred_logits =prob_atlas
            if interp:
                prob_atlas = F.interpolate(prob_atlas, size=shape_orig, mode='trilinear', align_corners=False)
                pred_logits = F.interpolate(pred_logits, size=shape_orig, mode='trilinear', align_corners=False)

        # return non-integrated flow field if training
        if not registration:
            return y_source, y_target, preint_flow, pos_flow, pred_logits, prob_atlas, prob_warped_input, y_source_smooth, prob_atlas_smooth
        else:
            return y_source, pos_flow, pred_logits, prob_atlas, prob_warped_input



