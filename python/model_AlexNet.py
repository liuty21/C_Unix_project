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

# assume input size = 28*28
class Model_Alexnet(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(Model_Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0), # 12*12*4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 6*6*4
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2), # 6*6*8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 3*3*8
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1), # 3*3*4
            nn.ReLU(inplace=True),
        )
        # output size = 3*3*4
        self.fc = nn.Sequential(
            nn.Linear(3*3*4, 18),
            nn.ReLU(inplace=True),
            nn.Linear(18, 10),
            )



    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(x.size(0), -1))
        return x