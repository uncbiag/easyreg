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

ffi = FFI()


class Bilinear(Function):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self):
        """
        Constructor
        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()

    def forward_stn(self, input1, input2, output, ndim, device_c):
        if ndim == 1:
            my_lib_1D.BilinearSamplerBCW_updateOutput_cuda_1D(input1, input2, output, device_c)
        elif ndim == 2:
            my_lib_2D.BilinearSamplerBCWH_updateOutput_cuda_2D(input1, input2, output, device_c)
        elif ndim == 3:
            my_lib_3D.BilinearSamplerBCWHD_updateOutput_cuda_3D(input1, input2, output, device_c)


    def backward_stn(self, input1, input2, grad_input1, grad_input2, grad_output, ndim, device_c):
        if ndim == 1:
            my_lib_1D.BilinearSamplerBCW_updateGradInput_cuda_1D(input1, input2, grad_input1, grad_input2,
                                                                 grad_output, device_c)
        elif ndim == 2:
            my_lib_2D.BilinearSamplerBCWH_updateGradInput_cuda_2D(input1, input2, grad_input1, grad_input2,
                                                                  grad_output, device_c)
        elif ndim == 3:
            my_lib_3D.BilinearSamplerBCWHD_updateGradInput_cuda_3D(input1, input2, grad_input1, grad_input2,
                                                                   grad_output, device_c)


    def forward(self, input1, input2):
        """
        Perform the actual spatial transform
        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        input1 = (input1+1)/2
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        self.ndim  =len(input1.size())-2
        if self.ndim == 1:
            output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2]).zero_()
        elif self.ndim == 2:
            output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3]).zero_()
        elif self.ndim == 3:
            output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3],input2.size()[4]).zero_()
        else:
            raise ValueError('Can only process dimensions 1-3')

        # print('decice %d' % torch.cuda.current_device())
        self.device = torch.cuda.current_device()
        self.device_c[0] = self.device
        self.forward_stn(input1, input2, output, self.ndim, self.device_c)
        return output*2-1

    def backward(self, grad_output):
        """
        Computes the gradient
        :param grad_output: grad output from previous "layer"
        :return: gradient
        """
        grad_output= grad_output*2.
        grad_input1 = torch.cuda.FloatTensor(self.input1.size()).zero_()
        grad_input2 = torch.cuda.FloatTensor(self.input2.size()).zero_()
        # print grad_output.view(1, -1).sum()
        # print('backward decice %d' % self.device)
        self.backward_stn(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.ndim, self.device_c)
        #print(torch.max(grad_input1), torch.max(grad_input2))
        return grad_input1 / 2., grad_input2

