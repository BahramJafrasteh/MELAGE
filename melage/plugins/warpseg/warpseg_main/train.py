# 1.10.0+cu111

# --modelName  Net3D_0_splt2none_dilated --state train --dataset threeD --GPUs 1 --batchSize 24 --shuffle --splitNumber 2 --modelNetG halfunet3d --lr 0.001 --timereg 0 --chkptDir chkpts --numThreads 12 --dilated
import torch
import sys
from skimage.measure import label as label_connector
def LargestCC(segmentation, connectivity=3):
    """
    Get largets connected components
    """
    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    return labels, frequency

def soft_dilation_3d(x, kernel_size=3):
    return F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def soft_erosion_3d(x, kernel_size=3):
    return -F.max_pool3d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

sys.path.append('../../')
import torch.nn as nn
from data_reader import get_parallel_dataloader, customLoader
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

# if dist.get_rank() == 0:
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
#parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--chkptDir', type=str, default='chkpts',
                    help='check points directory (Default chkpts)')
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batchSize', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--numThreads', type=int, default=12,
                    help='number of threads (default: 2)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--continueTraining', action='store_true',
                    help='if specified continue training from the last checkpint')
# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--shuffle', action='store_true',
                    help='if specified shuffle the training instances')
# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--state', default='train',
                    help='state train or test')
parser.add_argument('--modelName', type=str, default='', help='Automatic')

opt = parser.parse_args()




opt.chkptDir = os.path.join(opt.chkptDir, opt.modelName, 'newchkpt_1')

initial_seed = True
if initial_seed:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    seed = 0

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # torch.experimental.deterministic = True
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)


def save(model, optimizer, current_step, epoch='latest'):
    # if dist.get_rank() == 0:
    data = {
        'step': current_step,
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }


    name_model = 'New_model'
    save_filename = name_model + '_' + str(epoch) + '.pth'
    save_path = os.path.join(model.opt.chkptDir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_filepath = os.path.join(save_path, save_filename)
    # print('saving to {}'.format(save_filepath))
    torch.save(data, save_filepath)


def load(model, optimizer, rank, load_filepath, reset_optimizer=False):
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    #if 'cuda' in rank:
    #    data = torch.load(load_filepath, map_location=rank)
    #else:
    if is_integer(rank):
        data = torch.load(load_filepath, map_location=f"cuda:{rank}")
    else:
        data = torch.load(load_filepath, map_location=f"{rank}")

    try:
        keys = [data['model'][el].max() for el in data['model'].keys()]
        #for el in keys:
        #    print(el)
        #to_be_removed = [el for el in data['model'].keys() if 'seg_model' in el]
        #for el in to_be_removed:
        #    data['model'].pop(el)

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
        #missing_keys, unexpected_keys = model.load_state_dict(data['model'], strict=False)
        #print(missing_keys)
        #print(unexpected_keys)
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
    # try:
    #    lossSSIM.load_state_dict(data['loss'])
    # except:
    #    pass
    return current_step

def is_integer(val):
    try:
        int(val)
        return True
    except ValueError:
        return False

def plot_one_slice(imT1, prob_atlas,prob_mode, outfile):
    yy_model = ((prob_mode.argmax(1).unsqueeze(1) )).to(torch.float).detach().cpu().numpy()
    yy_atlas = ((prob_atlas.argmax(1).unsqueeze(1) )).to(torch.float).detach().cpu().numpy()
    im = imT1.detach().cpu().numpy().copy()
    fig, ax = plt.subplots(3, 3)
    ax[0,0].imshow(im[0, 0, :, :, im.shape[-1]//2-5, ], vmin=0, vmax=1)
    ax[0,0].set_title('T1')
    ax[0, 1].imshow(im[0, 0, :, :, im.shape[-1]//2, ])
    ax[0,1].set_title('T1')
    ax[0,2].imshow(im[0, 0, :, :, im.shape[-1]//2+5, ], vmin=0, vmax=1)
    ax[0, 2].set_title('T1')

    ax[1, 0].imshow((yy_model)[0, 0, :, :, im.shape[-1]//2-5, ], cmap='tab20')
    ax[1,0].set_title('Seg Ours')
    ax[1, 1].imshow((yy_model )[0, 0, :, :, im.shape[-1]//2, ], cmap='tab20')
    ax[1,1].set_title('Seg Ours')
    ax[1,2].imshow((yy_model)[0, 0, :, :, im.shape[-1]//2+5, ], cmap='tab20')
    ax[1,2].set_title('Seg Ours')


    ax[2, 0].imshow((yy_atlas)[0, 0, :, :, im.shape[-1]//2-5, ], cmap='tab20')
    ax[2,0].set_title('Seg Atlas')
    ax[2, 1].imshow((yy_atlas )[0, 0, :, :, im.shape[-1]//2, ], cmap='tab20')
    ax[2,1].set_title('Seg Atlas')
    ax[2,2].imshow((yy_atlas)[0, 0, :, :, im.shape[-1]//2+5, ], cmap='tab20')
    ax[2,2].set_title('Seg Atlas')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()




def train_model(rank, world_size):
    # torch.distributed.barrier()  # Ensure all processes are ready
    if world_size>1:
        setup_dist(rank, world_size)

    inshape = [224, 256, 192]#[192, 192, 192]

    model = vxm.networks.Multi_Segmentor(opt,
                                   inshape=inshape,tissue_ch=6
                                   ).to(rank)
    intermediate_folder = os.path.join(model.opt.chkptDir, 'intermediate_results')
    os.makedirs(intermediate_folder, exist_ok=True)
    for param in model.seg_model.parameters():
        param.requires_grad = True


    print('There are {} trainable parameters'.format(
        np.sum([par.numel() for par in model.parameters() if par.requires_grad])))

    image_loss_func = vxm.losses.NCC().loss

    # elif args.image_loss == 'mse':
    #image_loss_func = vxm.losses.MSE().loss
    image_loss_func_ssim = vxm.losses.SSIM3D_DIST(device=rank, window_size=3)
    image_loss_func_ssim.to(rank)
    image_loss_func_ssim.window.to(rank)



    image_loss_mse = vxm.losses.MultiScaleMSE()
    image_loss_mse.to(rank)

    kl_dice = vxm.losses.KLDiceLoss()
    kl_dice.to(rank)
    compute_bce = nn.BCEWithLogitsLoss()


    weights = [1]
    weights += [opt.weight]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    current_step = 0
    # model = DataParallel(model)
    name_model = 'New_model'
    if opt.continueTraining:
        # Load model weights if available

        #load_filename = name_model + '_latest_current' + '.pth'
        load_filename = name_model + '_latest_SegMulti_current' + '.pth'
        print('loading model from {}/{}'.format(opt.chkptDir, load_filename))
        current_step = load(model, optimizer, rank, load_filepath='{}/{}'.format(opt.chkptDir, load_filename),
                            reset_optimizer=False)
        model.to(rank)


        #model.unet_rec_t2.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if world_size>1:
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        ddp_model = model

    # Get dataloader
    if world_size>1:
        Train_dataloader = get_parallel_dataloader(opt, rank, world_size, 'train', pin_memory=True, prefetch_factor=2, name='segDN',
                                                   num_workers=4, collae_fn=None)
    else:
        Train_dataloader =  customLoader(opt, name='segDN', num_workers=4, prefetch_factor=2, collae_fn=None)

    opt.nEpoch = 4000
    iter = 0
    # Training loop
    use_loss_intensity = True
    for epoch in range(opt.nEpoch):

        epoch_total_loss = []

        epoch += current_step
        train_per_epoch = len(Train_dataloader)
        if rank == 0:# or ~is_integer(rank):
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=opt.nEpoch, width=50,
                              always_stateful=False)
            kbar.update(0, values=[("loss", 0),("Loss_total", 0), ("Loss_tissue", 0)])

        ddp_model.train()

        accumulation_steps = 1  # Number of batches to accumulate
        optimizer.zero_grad()

        for i, data in enumerate(Train_dataloader):
            # kbar.update(i + 1, values=[("loss", loss_show), ("RMSE", loss_show)])
            # continue

            imT1 = data['T1'].to(rank)
            prob_edt = data['T1_prob_edt'].to(rank)
            affine = data['affine']
            mask = data['maskT1'].to(rank)
            prob = data['T1_prob'].to(rank)
            weight_tissue = data['weight_tissue'].to(rank)
            target_tissue = data['prob_tissue'].to(rank)
            source_size = [el.item() for el in data['interShape']]
            #ventricles_edt = data['ventricles_edt'].to(rank)
            #ventricles_3_whole = data['dist_vent3_whole'].to(rank)
            #ventricles_19_whole = data['dist_vent19_whole'].to(rank)

            #source_size = list(source_size.squeeze().detach().cpu().numpy())


            pred_logits, pred_logits_tissue = model(imT1, source_size)


            loss = 0

            loss_seg = vxm.losses.compute_segmentation_loss(pred_logits, prob, prob_edt)
            loss_tissue = vxm.losses.compute_segmentation_loss(pred_logits_tissue, target_tissue, weight_tissue)  # (Define similarly)
            loss = loss_seg + loss_tissue
            epoch_total_loss.append(loss.item())

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            model.sigma.data = torch.clamp(model.sigma.data, min=0.01, max=0.2)
            bsnm = os.path.basename(data['pathA'][0])
            basename = os.path.join(intermediate_folder, 'seg_'+bsnm[:bsnm.find('.')] + '.jpg')

            #if iter %1000 ==0:
            #    plot_one_slice(imT1, prob, prob_value, basename)

            if rank == 0:# or ~is_integer(rank):

                #kbar.update(i + 1, values=[("loss", loss.item())])
                #kbar.update(i + 1, values=[("loss", loss.item()), ("SSIM", im_loss_ssim.item()), ("NCC", im_loss_ncc.item()),
                #                       ('Jac', jac_penalty.item()), ('Grad', mse_img.item()), ('Bending', mse_frequ.item())])
                kbar.update(i + 1, values=[("loss", loss.item()), ("Loss_total", loss_seg.item()),
                                       ('Loss_tissue', loss_tissue.item())])


            torch.cuda.empty_cache()
            iter+=1
            if iter%1000==0 and rank==0:
                print('saving model at iter {}\n'.format(iter))
                if world_size > 1:
                    save(ddp_model.module, optimizer, epoch, 'Segintermediate_iteration_{}'.format(iter))
                else:
                    save(ddp_model, optimizer, epoch, 'Segintermediate_iteration_{}'.format(iter))
        if epoch % 1000 == 0:
            save(ddp_model.module, optimizer, epoch, 'latest_Seg{}'.format(epoch))
        else:
            if world_size > 1:
                save(ddp_model.module, optimizer, epoch, 'latest_SegMulti_current')
            else:
                save(ddp_model, optimizer, epoch, 'latest_SegMulti_current')
        path_save = os.path.join(opt.chkptDir, 'save_images')
        path_save = os.path.join(path_save, "DS_{}.png".format(epoch))


    save(ddp_model.module, optimizer, 0, 'latestSegMulti')
    cleanup()


# testDataLoader = customLoader(opt, 'test', shuffle=False)


def main():
    try:
        world_size = torch.cuda.device_count()  # Number of GPUs to use
        print(world_size)
        #world_size = 1
        if world_size > 0:
            mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)
        else:
            train_model(rank='cuda:0', world_size=1)
    except:
        train_model(rank='cpu', world_size=1)


if __name__ == "__main__":
    main()






