import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from torch.autograd import Variable

import torch
import copy

def np2gputensor(data):
    data = torch.FloatTensor(data)
    data = Variable(data)
    data = data.cuda()
    return data

def tensor2numpy(tensorvar):
    if tensorvar.is_cuda or tensorvar.requires_grad:
        return copy.deepcopy(tensorvar.detach().cpu().numpy())
    else:
        return copy.deepcopy(tensorvar.numpy())

