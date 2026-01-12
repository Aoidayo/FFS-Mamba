import os
import numpy as np
import random
import torch

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def cal_model_size(model):
    """ Calculate the total learnable parameter size (in megabytes) of a torch module. """
    trainable_param_size = 0
    total_param_size = 0
    for param in model.parameters():
        total_param_size += param.nelement() * param.element_size()
        if param.requires_grad: # trainable
            trainable_param_size += param.nelement() * param.element_size()

    trainable_size_all_mb = trainable_param_size / 1024**2
    total_size_all_mb = total_param_size / 1024**2
    return trainable_size_all_mb, total_size_all_mb