from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

class DIRNet(nn.Module):
    def __init__(self, opt):
        super(DIRNet,self).__init__()