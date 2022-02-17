import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.plugin import PruningNetworkPlugin
from utils.mask_op import MaskManager_skip
from torch.autograd import Variable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class DownsampleA(nn.Module):  
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__() 
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

    def forward(self, x):   
        x = self.avg(x)  
        return torch.cat((x, x.mul(0)), 1)  

class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
    
        return F.relu(residual + basicblock, inplace=True)

class CifarResNet(PruningNetworkPlugin):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes):
        """ Constructor
        Args:
            depth: number of layers.
            num_classes: number of classes
        """
        super().__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        self.layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, self.layer_blocks))

        self.block = block
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask_conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(16, 1)
        self.stage_2 = self._make_layer(32, 2)
        self.stage_3 = self._make_layer(64, 2)
        
        self.inplanes = 16
        self.mask_stage_1 = self._make_layer(16, 1)
        self.mask_stage_2 = self._make_layer(32, 2)
        self.mask_stage_3 = self._make_layer(64, 2)
        
        # init maskman
        self.maskman = MaskManager_skip()
        
        # add maskman conv_1_3x3
        self.maskman.add(key='conv_1_3x3',origin=self.conv_1_3x3,mask=self.mask_conv_1_3x3,input_size=[3,32,32],output_size=[16,32,32])
        
        # add maskman stage_1
        for i in range(self.layer_blocks):
            planes = [16,32,32]
            if i==0 : in_planes = [16,32,32]
            else : in_planes = [16,32,32]
            self.maskman.add(key='conv1_{}_a'.format(i),origin=self.stage_1[i].conv_a,mask=self.mask_stage_1[i].conv_a,input_size=in_planes,output_size=planes)
            self.maskman.add(key='conv1_{}_b'.format(i),origin=self.stage_1[i].conv_b,mask=self.mask_stage_1[i].conv_b,input_size=planes,output_size=planes)
        
        # add maskman stage_2
        for i in range(self.layer_blocks):
            planes = [32,16,16]
            if i==0 : in_planes = [16,32,32]
            else : in_planes = [32,16,16]
            self.maskman.add(key='conv2_{}_a'.format(i),origin=self.stage_2[i].conv_a,mask=self.mask_stage_2[i].conv_a,input_size=in_planes,output_size=planes)
            self.maskman.add(key='conv2_{}_b'.format(i),origin=self.stage_2[i].conv_b,mask=self.mask_stage_2[i].conv_b,input_size=planes,output_size=planes)
        
        # add maskman stage_3
        for i in range(self.layer_blocks):
            planes = [64,8,8]
            if i==0 : in_planes = [32,16,16]
            else : in_planes = [64,8,8]
            self.maskman.add(key='conv3_{}_a'.format(i),origin=self.stage_3[i].conv_a,mask=self.mask_stage_3[i].conv_a,input_size=in_planes,output_size=planes)
            self.maskman.add(key='conv3_{}_b'.format(i),origin=self.stage_3[i].conv_b,mask=self.mask_stage_3[i].conv_b,input_size=planes,output_size=planes)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64, num_classes)
        self.mask_classifier = nn.Linear(64, num_classes)
        
        # add maskman classifier
        self.maskman.add(key='fc',origin=self.classifier,mask=self.mask_classifier,input_size=[64],output_size=[10])
       
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        
        # init mask weight to ones
        self.maskman.initialize_mask()
    
    def _make_layer(self, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = DownsampleA(self.inplanes, planes, stride)

        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, self.layer_blocks):
            layers.append(self.block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.maskman.masking()
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def test(self):
        print("===========================================")
        print("Model layer by layer result")
        print("===========================================")
        self.test_state_dict()

def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    model.maskman.arch = 'resnet20'
    model.maskman.init()
    return model

def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    model.maskman.arch = 'resnet32'
    model.maskman.init()
    return model

def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    model.maskman.arch = 'resnet56'
    model.maskman.init()
    return model
