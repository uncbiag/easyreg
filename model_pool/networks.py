import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from .network_pool import ResnetGenerator, UnetGenerator

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_first_net(input_nc, output_nc, ngf, which_model_net_f, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netf = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_net_f == 'resnet_9blocks':
        netf = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_f == 'resnet_6blocks':
        netf = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_f == 'unet_128':
        netf = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_f== 'unet_256':
        netf = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_net_f)
    if len(gpu_ids) > 0:
        netf.cuda(device_id=gpu_ids[0])
    netf.apply(weights_init)
    return netf


def define_second_net(input_nc, output_nc, ngf, which_model_net_s, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)


    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_net_s == 'resnet_9blocks':
        net_s = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_s == 'resnet_6blocks':
        net_s = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_s == 'unet_128':
        net_s = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_net_s== 'unet_256':
        net_s = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_net_f)
    if len(gpu_ids) > 0:
        net_s.cuda(device_id=gpu_ids[0])
    net_s.apply(weights_init)
    return net_s


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



