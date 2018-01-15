from cffi import FFI
import torch
from torch.autograd import Variable

ffi = FFI()

from lib._ext import nn_interpolation

def nn_interpolation_fn_sel(input1, input2, output, ndim, device_c):
    if ndim == 1:
        nn_interpolation.nearestNeighBCW_updateOutput_cuda_1D(input1, input2, output, device_c)
    elif ndim == 2:
        nn_interpolation.nearestNeighBCWH_updateOutput_cuda_2D(input1, input2, output, device_c)
    elif ndim == 3:
        nn_interpolation.nearestNeighBCWHD_updateOutput_cuda_3D(input1, input2, output, device_c)



def get_nn_interpolation(input1, input2):
    device_c = ffi.new("int *")
    ndim = len(input1.size())-2
    if ndim == 1:
        output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2]).zero_()
    elif ndim == 2:
        output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3]).zero_()
    elif ndim == 3:
        output = torch.cuda.FloatTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3], input2.size()[4]).zero_()
    else:
        raise ValueError('Can only process dimensions 1-3')
    # print('decice %d' % torch.cuda.current_device())
    device_c[0] = torch.cuda.current_device()
    nn_interpolation_fn_sel(input1.data, input2.data, output, ndim, device_c)
    return Variable(output)
