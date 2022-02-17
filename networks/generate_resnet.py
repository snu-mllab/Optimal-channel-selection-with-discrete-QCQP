import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from torch.autograd import Variable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes[0], stride)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes[0], planes[1])
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            self.residual = residual
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.conv3 = nn.Conv2d(planes[1], planes[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes[2])
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, ch_info, num_classes=1000):
        self.idx = 0
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, ch_info[self.idx], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(ch_info[self.idx])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ch_info, layers[0])
        self.layer2 = self._make_layer(block, ch_info, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ch_info, layers[2], stride=2)
        self.layer4 = self._make_layer(block, ch_info, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(ch_info[self.idx], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, ch_info, blocks, stride=1):
        downsample = None
        if block.expansion==1:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.Conv2d(ch_info[self.idx], ch_info[self.idx+2],
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(ch_info[self.idx+2]),
                )

            layers = []
            layers.append(block(ch_info[self.idx], ch_info[self.idx+1:self.idx+3], stride, downsample))
            for i in range(1, blocks):
                self.idx += 2
                layers.append(block(ch_info[self.idx], ch_info[self.idx+1:self.idx+3]))
            self.idx += 2
        elif block.expansion==4:
            downsample = nn.Sequential(
                nn.Conv2d(ch_info[self.idx], ch_info[self.idx+3],
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_info[self.idx+3]),
            )

            layers = []
            layers.append(block(ch_info[self.idx], ch_info[self.idx+1:self.idx+4], stride, downsample))
            for i in range(1, blocks):
                self.idx += 3
                layers.append(block(ch_info[self.idx], ch_info[self.idx+1:self.idx+4]))
            self.idx += 3
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def gen_resnet50(pretrained=False, ch_info=[64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048, 1000], **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        ch_info : list of int. 
            list of outch of each layer.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], ch_info, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
        print('ResNet-50 Use pretrained model for initalization')
    return model
