import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.pytorch_op import get_num_param, get_zero_param

import torch
from torch import nn


class NetworkPlugin(nn.Module):
    def __init__(self):
        super(NetworkPlugin, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def test_state_dict(self):
        print("============================================")
        print("Model's state_dict")
        print("============================================")
        for param_tensor in self.state_dict():
            print("{} \t {}".format(param_tensor, self.state_dict()[param_tensor].size()))

    def get_trainable_parameters(self):
        return [w for name, w in self.named_parameters() if not 'mask' in name]

    def save(self, path):
        print("============================================")
        print("Model is saved in {}".format(path))
        print("============================================")
        torch.save(self.state_dict(), path)

    def load_custom(self, path, map_location=None):
        print("============================================")
        print("Model is loaded from {}".format(path))
        print("============================================")

        if map_location is None: 
            load_dict = torch.load(path)
        else:
            load_dict = torch.load(path, map_location=map_location)
        return load_dict

    def load(self, path, map_location=None, key_changer=None):
        load_dict = self.load_custom(path=path, map_location=map_location)
        self.load_from_state_dict(source_dict=load_dict, key_changer=key_changer)

    def load_from_state_dict(self, source_dict, key_changer=None):
        if key_changer is None:
            self.load_state_dict({key: value for key, value in source_dict.items() if key in self.state_dict()}, strict=False)
        else:
            self.load_state_dict({key_changer(key): value for key, value in source_dict.items() if key_changer(key) in self.state_dict()}, strict=False)
    
    def load_mask(self, path, map_location=None):
        load_dict = self.load_custom(path=path, map_location=map_location)
        self.load_mask_from_state_dict(source_dict=load_dict)

    def load_non_mask(self, path, map_location=None):
        load_dict = self.load_custom(path=path, map_location=map_location)
        self.load_non_mask_from_state_dict(source_dict=load_dict)

    def load_mask_from_state_dict(self, source_dict):
        source_dict = {key: value for key, value in source_dict.items() if 'mask' in key} 
        self.load_state_dict({key: value for key, value in source_dict.items() if key in self.state_dict()}, strict=False)

    def load_non_mask_from_state_dict(self, source_dict):
        source_dict = {key: value for key, value in source_dict.items() if 'mask' not in key} 
        self.load_state_dict({key: value for key, value in source_dict.items() if key in self.state_dict()}, strict=False)

class PruningNetworkPlugin(NetworkPlugin):
    def __init__(self):
        super().__init__()

    def get_mask_parameters(self):
        return [w for name, w in self.named_parameters() if 'mask' in name]

    def get_trainable_parameters(self):
        print("train : {}".format(len([name for name, w in self.named_parameters() if not 'mask' in name])))
        return [w for name, w in self.named_parameters() if not 'mask' in name]

    def get_conv_parameters(self):
        return [w for name, w in self.named_parameters() if ('mask' not in name) and ('conv' in name)]

    def get_fc_parameters(self):
        return [w for name, w in self.named_parameters() if ('mask' not in name) and ('fc' in name)]

    def get_zero_rate(self, target='all'):
        if target=='all':
            total_count = get_num_param([w for w in self.get_trainable_parameters()])
            zero_count = get_zero_param([w for w in self.get_trainable_parameters()])
        elif target=='conv':
            total_count = get_num_param([w for w in self.get_conv_parameters()])
            zero_count = get_zero_param([w for w in self.get_conv_parameters()])
        elif target=='fc':
            total_count = get_num_param([w for w in self.get_fc_parameters()])
            zero_count = get_zero_param([w for w in self.get_fc_parameters()])
        else:
            assert False, "Invalid target : {}".format(target)
        return zero_count/total_count

