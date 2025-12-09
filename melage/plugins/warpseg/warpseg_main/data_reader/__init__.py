import numpy as np
import torch
import torch.nn.functional as F
from .DDSet import DDSet
from .DDSetSeg import DDSetSeg
import argparse
from torch.utils.data.distributed import DistributedSampler
# function loaders
funcLoaders = {'D':DDSet,'segDN':DDSetSeg}
import os
from argparse import Namespace
import os
import torch
from argparse import Namespace

def get_dataloader(opt, split='train', pin_memory=True, prefetch_factor=2,
                   name='D', num_workers=None, collate_fn=None, params=None):
    # Use a shallow copy, as we are only modifying top-level attributes.
    # If opt contained nested dicts/lists that you intended to modify,
    # you would use opt_local = copy.deepcopy(opt)
    opt_local = Namespace(**vars(opt))

    # 1. Use a descriptive name like 'split' instead of 'type'
    opt_local.state = split.lower()
    is_train = (opt_local.state == 'train')

    # 2. Set batch size for non-training splits
    if not is_train:
        opt_local.batchSize = 1

    dataset = funcLoaders[name]()
    dataset.options(opt_local, params, None)

    # 3. Use a more sensible default for num_workers
    if num_workers is None:
        # Default to 4 workers, or all CPUs if less than 4 are available
        num_workers = min(os.cpu_count(), 4)

    print(f"Using {num_workers} workers for the '{split}' split.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt_local.batchSize,
        # 4. Correctly set shuffle only for training
        shuffle=is_train,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        # 5. It's common to drop the last batch for training
        drop_last=is_train,
        pin_memory=pin_memory,
        # 6. Fix typo
        collate_fn=collate_fn
    )

    return dataloader



if __name__=='__main__':
    None
