import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralConv(nn.Module):
    def __init__(self, nin, nout, nker, pad, stride, bias=False):
        super(GeneralConv , self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(nin, nout, kernel_size=nker, padding=pad, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        print("nn.Conv2d({}, {}, kernel_size={}, padding={}, stride={}, bias={})".format(nin, nout, nker, pad, stride, bias))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class BasicConv(nn.Module):
    def __init__(self, nin, nout, nker, pad, stride, bias=False):
        super(BasicConv , self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=nker, padding=pad, stride=stride, bias=bias)
        print("nn.Conv2d({}, {}, kernel_size={}, padding={}, stride={}, bias={})".format(nin, nout, nker, pad, stride, bias))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        return x


