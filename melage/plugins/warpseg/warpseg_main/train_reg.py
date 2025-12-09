import torch
import sys

sys.path.append('../../')
import torch.nn as nn
from data_reader import get_dataloader
from dist_utils import setup_dist, cleanup
import os
os.environ['VXM_BACKEND'] = 'pytorch'
import torch.multiprocessing as mp
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import voxelmorph as vxm
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import time
import pkbar
import random
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.distributed as dist

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters



parser.add_argument('--chkptDir', type=str, default='chkpts',
                    help='check points directory (Default chkpts)')
# training parameters

parser.add_argument('--batchSize', type=int, default=1, help='batch size (default: 1)')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

parser.add_argument('--continueTraining', action='store_true',
                    help='if specified continue training from the last checkpint')
# network architecture parameters
parser.add_argument('--shuffle', action='store_true',
                    help='if specified shuffle the training instances')
# loss hyperparameters

parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')

parser.add_argument('--modelName', type=str, default='', help='Automatic')

args = parser.parse_args()




args.chkptDir = os.path.join(args.chkptDir, args.modelName, 'newchkpt_1')

initial_seed = True
if initial_seed:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    seed = 0

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)


    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)


def save(model, optimizer, current_step, epoch='latest'):
    """
    Save the model checkpoint.
    """

    data = {
        'step': current_step,
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }


    name_model = 'New_model'
    save_filename = name_model + '_' + str(epoch) + '.pth'
    save_path = os.path.join(model.args.chkptDir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_filepath = os.path.join(save_path, save_filename)

    torch.save(data, save_filepath)


def load(model, optimizer, rank, load_filepath, reset_optimizer=False):
    """
    Load the model checkpoint.
    """

    if is_integer(rank):
        data = torch.load(load_filepath, map_location=f"cuda:{rank}")
    else:
        data = torch.load(load_filepath, map_location=f"{rank}")

    try:
        keys = [data['model'][el].max() for el in data['model'].keys()]

        model_state = model.state_dict()
        pretrained_state = data["model"]

        # Filter out keys where shape doesn't match
        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }

        # Update the current model's state_dict
        model_state.update(filtered_state)

        # Load the updated state_dict
        model.load_state_dict(model_state)

    except:
        try:
            model.load_state_dict(data['state_dict'], strict=False)
        except:
            print('Unable to load model')

    current_step = 0
    if reset_optimizer:
        try:
            optimizer.load_state_dict(data['opt'])
            current_step = data['step']
        except:
            pass
    try:
        current_step = data['step']
    except:
        pass

    return current_step

def is_integer(val):
    try:
        int(val)
        return True
    except ValueError:
        return False

def plot_one_slice(image1, image2, outfile):
    """
    Plot a middle slice of two 3D images side by side.
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow((image1.detach().cpu().numpy())[0, 0, image2.shape[2] // 2, :, :])

    ax[1].imshow(((image2).detach().cpu().numpy())[0, 0, image2.shape[2] // 2, :, :])
    plt.savefig(outfile)
    plt.close()

def train_model(rank, world_size):
    """
    Main training function.
    """
    if world_size>1:
        setup_dist(rank, world_size)

    encoder_filters = [16, 32, 32, 32]
    decoder_filters = [32, 32, 32, 32, 32, 16, 16]
    inshape = [224, 256, 192]
    integration_steps = 7
    integration_downsize = 2

    model = vxm.networks.FVxmDense(args,
            inshape=inshape,
            nb_unet_features=[encoder_filters, decoder_filters],
            bidir=False,
            int_steps=integration_steps,
            int_downsize=integration_downsize
        ).to(rank)
    intermediate_folder = os.path.join(model.args.chkptDir, 'intermediate_results')
    os.makedirs(intermediate_folder, exist_ok=True)
    for param in model.parameters():
        param.requires_grad = True
    for param in model.flow.parameters():
        param.requires_grad = True
    for param in model.seg_model.parameters():
        param.requires_grad = False

    print('There are {} trainable parameters'.format(
        np.sum([par.numel() for par in model.parameters() if par.requires_grad])))

    image_loss_func = vxm.losses.NCC().loss

    image_loss_func_ssim = vxm.losses.SSIM3D_DIST(device=rank, window_size=3)
    image_loss_func_ssim.to(rank)
    image_loss_func_ssim.window.to(rank)
    match_histogram = vxm.torch.utils.Histogram_Matching()
    match_histogram.to(rank)
    image_loss_mse = vxm.losses.MultiScaleMSE()
    image_loss_mse.to(rank)

    kl_dice = vxm.losses.KLDiceLoss()
    kl_dice.to(rank)

    perceptual_loss = vxm.losses.PerceptualLossMedNet3D(device=rank)

    grad_loss = vxm.losses.Grad('l2', loss_mult=integration_downsize).loss
    weights = [1]
    weights += [args.weight]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    current_step = 0

    name_model = 'New_model'
    if args.continueTraining:
        # Load model weights if available

        load_filename = name_model + '_latest_current' + '.pth'
        print('loading model from {}/{}'.format(args.chkptDir, load_filename))
        current_step = load(model, optimizer, rank, load_filepath='{}/{}'.format(args.chkptDir, load_filename),
                            reset_optimizer=False)
        model.to(rank)
        model.unet_model.to(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if world_size>1:
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        ddp_model = model

    # Get dataloader
    Train_dataloader = get_dataloader(args, 'train', pin_memory=True, prefetch_factor=2,
                                                    name='D')


    args.nEpoch = 4000
    iter = 0
    # Training loop
    use_loss_intensity = True
    use_registration = True
    for epoch in range(args.nEpoch):

        epoch_total_loss = []

        epoch += current_step
        train_per_epoch = len(Train_dataloader)
        if rank == 0:# or ~is_integer(rank):

            progress_bar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=args.nEpoch, width=50,
                               always_stateful=False)
            if use_registration:
                progress_bar.update(0, values=[("loss", 0.0), ("SSIM", 0.0), ("NCC", 0.0),
                                   ('Jac', 0.0), ('Grad', 0.0)])
            else:
                progress_bar.update(0, values=[("loss", 0), ("SoftDICE", 0),
                                       ("MSE", 0),
                                       ('Entropy', 0), ('TotalVariation', 0)])

        ddp_model.train()

        accumulation_steps = 1  # Number of batches to accumulate
        optimizer.zero_grad()

        for i, data in enumerate(Train_dataloader):

            t1_image = data['T1'].to(rank)

            t1_mask = data['maskT1'].to(rank)
            t1_mask_no_csf = data['maskNoCSF'].to(rank)
            atlas_image = data['atlas'].to(rank)
            atlas_probabilities = data['atlas_probs'].to(rank)

            image_path = data['pathA'][0]

            if use_registration:
                warped_atlas, pre_flow, pos_flow, pred_logits, prob_atlas = model(atlas_image, t1_image, t1_mask,
                                                                                     atlas_probabilities, data_info=data, mask_noCSF=t1_mask_no_csf)
            else:
                model(atlas_image, t1_image, t1_mask,
                                                                                     atlas_probabilities, data_info=data, mask_noCSF=t1_mask_no_csf)
                continue

            loss = 0

            jacobian_magnitude = vxm.losses.compute_jacobian_determinant(pos_flow).unsqueeze(1)

            eroded = vxm.torch.utils.erode_mask(t1_mask, radius=2)
            jacobian_magnitude = jacobian_magnitude * eroded

            loss_voxelmorph = 0
            im_loss_ncc = image_loss_func(t1_image*t1_mask, warped_atlas*t1_mask)/2.0
            loss_voxelmorph += im_loss_ncc
            im_loss_ssim = image_loss_func_ssim(t1_image*t1_mask, warped_atlas*t1_mask)
            loss_voxelmorph += im_loss_ssim
            loss_laplace = 0
            jac_penalty = (torch.abs(1-jacobian_magnitude) ** 2).mean()
            loss_voxelmorph += 0.5*jac_penalty
            mse_img = ((t1_image * t1_mask - warped_atlas*t1_mask)).abs().mean()
            loss_voxelmorph += mse_img
            grad_= 0.01*grad_loss(t1_image*t1_mask, pre_flow)
            loss_voxelmorph += grad_
            loss += loss_voxelmorph

            epoch_total_loss.append(loss.item())

            loss.backward(retain_graph=False)

            optimizer.step()
            optimizer.zero_grad()
            model.sigma.data = torch.clamp(model.sigma.data, min=0.01, max=0.2)
            basename = os.path.basename(data['pathA'][0])
            basename = os.path.join(intermediate_folder, basename[:basename.find('.')] + '.jpg')

            if rank == 0:

                if use_registration:
                    progress_bar.update(i + 1, values=[("loss", loss.item()), ("SSIM", im_loss_ssim.item()), ("NCC", im_loss_ncc.item()),
                                       ('Jac', jac_penalty.item()), ('Grad', mse_img.item())])
                else:
                    progress_bar.update(i + 1, values=[("loss", loss.item()), ("SoftDICE", softDiceloss.item()), ("MSE", weighted_mse_loss.item()),
                                       ('Entropy', entropy.item()), ('TotalVariation', tv.item())])


            torch.cuda.empty_cache()
            iter+=1
            if iter%1000==0 and rank==0:
                print('saving model at iter {}\n'.format(iter))
                save(ddp_model.module, optimizer, epoch, 'intermediate_iteration_{}'.format(iter))
        if epoch % 10 == 0:
            save(ddp_model.module, optimizer, epoch, 'latest_{}'.format(epoch))
        else:
            save(ddp_model.module, optimizer, epoch, 'latest_current')
        path_save = os.path.join(args.chkptDir, 'save_images')
        path_save = os.path.join(path_save, "DS_{}.png".format(epoch))


    save(ddp_model.module, optimizer, 0, 'latest')
    cleanup()





def main():
    world_size = torch.cuda.device_count()  # Number of GPUs to use
    print(world_size)

    if world_size > 0:
        mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)
    else:
        train_model(rank='cuda:0', world_size=1)


if __name__ == "__main__":
    main()






