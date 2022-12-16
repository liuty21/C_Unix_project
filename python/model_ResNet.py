import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

class residual(nn.Module):
    """docstring for residual"""
    def __init__(self, in_channel, out_channel, stride=1, transform=0):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias = False)
        self.transform = transform # type=0: direct identity connect; type=1:need transformation
        # transformation for identity connection
        self.conv_identity = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        identity = x
        if self.transform == 1:
            identity = self.conv_identity(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out
        

# assume input size = 28*28
class ResNet(nn.Module):
    """docstring for ResNet"""
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0) # 12*12*4
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 6*6*4
        self.res1 = residual(4, 4, 1, 0) # 6*6*4
        self.res2 = residual(4, 8, 2, 1) # 3*3*8
        
        self.fc = nn.Sequential(
            nn.Linear(3*3*8, 18),
            nn.ReLU(inplace=True),
            nn.Linear(18, 10),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.res1(out)
        out = self.res2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
        