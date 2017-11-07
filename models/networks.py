from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from models.modules import *
from functions.bilinear import *
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.dense_gen = DisGen()
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        disField = self.dense_gen(input)
        output = self.bilinear(moving,disField)
        return output
