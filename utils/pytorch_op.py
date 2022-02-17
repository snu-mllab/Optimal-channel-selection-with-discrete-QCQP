import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch 
import numpy as np 

from utils.pytorch_img_op import tensor2numpy

def get_num_param(params_list):
    count = 0 
    for w in params_list:
        if w.is_cuda:
            w=w.cpu().detach()
        tmp = 1
        for v in list(w.data.size()):
            tmp*=v
        count+=tmp
    return count

def get_zero_param(params_list):
    count = 0
    for w in params_list:
        if w.is_cuda:
            w=w.cpu().detach()
        count+=torch.sum(torch.eq(w, torch.zeros(w.size())).float())
    return count

def update_target(target, new_value, mode="soft"):
    if target.is_cuda:
        device = target.get_device()
    else:
        device = 'cpu'

    if mode=="hard":
        if torch.is_tensor(new_value):
            target.data = torch.FloatTensor(tensor2numpy(new_value)).to(device)
        else:
            target.data = torch.FloatTensor(new_value).to(device)

    if torch.is_tensor(new_value):
        target.data = torch.mul(target, new_value)
    else:
        target.data = torch.FloatTensor(np.multiply(new_value, tensor2numpy(target.data))).to(device)

def init_module_weight_2_one(module):
    '''
    Args : 
        module => example nn.linear( ), nn.Conv2d( )
    '''
    if module.weight.is_cuda:
        device = module.weight.get_device()
    else:
        device = 'cpu'
    module.weight.data = torch.ones(module.weight.size()).to(device)

