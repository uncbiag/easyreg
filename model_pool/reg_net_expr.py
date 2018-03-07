from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from model_pool.modules import *
from functions.bilinear import *


class SimpleNet(nn.Module):
    def __init__(self, img_sz=None):
        super(SimpleNet,self).__init__()
        self.img_sz = img_sz
        self.denseGen = DisGen()
        self.jacobiField = JacobiField()
        self. denseAffineGrid= DenseAffineGridGen(self.img_sz)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        disField = self.denseGen(input)
        jacobDisField = self.jacobiField(disField)
        gridField = self.denseAffineGrid(disField)
        #gridField= torch.tanh(gridField)
        output = self.bilinear(moving,gridField)
        return output, gridField, disField



class FlowNet(nn.Module):
    def __init__(self, img_sz, spacing):
        super(FlowNet,self).__init__()
        self.img_sz = img_sz
        self.spacing = spacing
        self.momConv = MomConv(bn=False)
        self.jacobiField = JacobiField()
        self.flowRnn= FlowRNN(self.img_sz,self.spacing, bn=False)
        self.grid = grid_gen(img_sz)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        x = self.momConv(input[0], input[1])
        gridField, disField = self.flowRnn(self.grid,x, n_time=5)
        jacobDisField = self.jacobiField(disField)
        output = self.bilinear(moving,gridField)
        return output, gridField,disField
