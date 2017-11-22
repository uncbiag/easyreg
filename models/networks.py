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
    def __init__(self, info):
        super(SimpleNet,self).__init__()
        self.info = info
        self.denseGen = DisGen()
        self.jacobiField = JacobiField()
        self. denseAffineGrid= DenseAffineGridGen(self.info)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        disField = self.denseGen(input)
        jacobDisField = self.jacobiField(disField)
        gridField = self.denseAffineGrid(disField)
        output = self.bilinear(moving,gridField)
        return output, jacobDisField



class FlowNet(nn.Module):
    def __init__(self, info):
        super(SimpleNet,self).__init__()
        self.info = info
        self.momConv = MomConv()
        self.jacobiField = JacobiField()
        self.flowRnn= FlowRNN(self.info, bn=True)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        x = self.momConv(input[0], input[1])
        gridField, disField = self.flowRnn(x)
        jacobDisField = self.jacobiField(disField)
        output = self.bilinear(moving,gridField)
        return output, jacobDisField
