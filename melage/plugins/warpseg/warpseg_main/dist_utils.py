import os
import torch.distributed as dist
import torch
def setup_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Set this to the address of the master node
    os.environ['MASTER_PORT'] = '12353'       # Set this to an available port
    #dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    dist.init_process_group(
        backend="nccl",          # Use 'nccl' for GPUs, 'gloo' for CPU
        init_method="env://",    # URL specifying how to initialize (e.g., environment variables)
        world_size=world_size,   # Total number of processes
        rank=rank                # Rank of the current process
    )
    torch.cuda.set_device(rank)  # Set the current GPU device


def cleanup():
    dist.destroy_process_group()