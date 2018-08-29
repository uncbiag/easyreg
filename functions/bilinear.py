"""
Spatial transform functions in 1D, 2D, and 3D.
.. todo::
    Add CUDA implementation. Could be based of the existing 2D CUDA implementation.
"""

import torch
from torch.autograd import Function
from lib._ext import  my_lib_1D,my_lib_2D,my_lib_3D
from cffi import FFI
from torch.autograd import Variable
from torch.nn import Module

ffi = FFI()


class Bilinear(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]),
                                                     padding_mode=self.zero_boundary)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """


        output = self.forward_stn((input1+1)/2, input2)
        # print(STNVal(output, ini=-1).sum())
        return output*2-1
