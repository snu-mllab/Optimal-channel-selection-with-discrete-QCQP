import torch
import random

import numpy as np

def device_test():
    print("Current device : {}".format(torch.cuda.current_device()))
    print("Device count : {}".format(torch.cuda.device_count()))
    print("Cuda available : {}".format(torch.cuda.is_available()))

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_use_device_info():
    device_test()
    device = get_device()
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,4), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,4), 'GB')

def fix_random(rseed):
    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

