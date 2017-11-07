from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import *


class ControlGen(nn.Module):
    def __init__(self, n_control, bn=False, use_spp=False, spp_levels=(8,16)):
        super(ControlGen, self).__init__()

        # Build a LSTM
        self.conv1 = nn.Sequential(ConvBnRel(3, 64, 3, active_unit='elu', same_padding=False, bn=bn),
                                   nn.AvgPool2d(2))
        self.conv2 = nn.Sequential(ConvBnRel(64, 128, 3, active_unit='elu', same_padding=False, bn=bn),
                                   ConvBnRel(128, 128, 3, active_unit='elu', same_padding=False, bn=bn),
                                   nn.AvgPool2d(2))
        self.conv3 = nn.Sequential(ConvBnRel(128, 2, 3, active_unit='elu', same_padding=False, bn=bn))
        self.use_spp = use_spp
        if use_spp:
            self.spp_levels = spp_levels
            self.spp = SPPLayer(spp_levels, pool_type='average_pool')
            spp_dim = 0
            for i in spp_levels:
                spp_dim += i*i
            self.fc1 = FcRel(spp_dim, n_control, active_unit='None')
        else:
            self.fc1 = FcRel(None, n_control, active_unit='None')


    def forward(self, im_data):
        # im_data, im_scales = get_blobs(image)
        # im_info = np.array(
        #     [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        #     dtype=np.float32)
        # data = Variable(torch.from_numpy(im_data)).cuda()
        # x = data.permute(0, 3, 1, 2)

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.use_spp:
            x= self.ssp(x)
        x = self.fc1(x)

        return x


class DisGen(nn.Module):
    def __init__(self, bn=False):
        super(DisGen, self).__init__()

        # Build a LSTM
        self.conv1 = nn.Sequential(ConvBnRel(3, 64, 3, active_unit='elu', same_padding=True, bn=bn))
        self.conv2 = nn.Sequential(ConvBnRel(64, 128, 3, active_unit='elu', same_padding=True, bn=bn),
                                   ConvBnRel(128, 128, 3, active_unit='elu', same_padding=True, bn=bn))
        self.conv3 = nn.Sequential(ConvBnRel(128, 2, 3, active_unit='elu', same_padding=True, bn=bn))



    def forward(self, im_data):
        # im_data, im_scales = get_blobs(image)
        # im_info = np.array(
        #     [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        #     dtype=np.float32)
        # data = Variable(torch.from_numpy(im_data)).cuda()
        # x = data.permute(0, 3, 1, 2)

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)

        return x





class SPPLayer(nn.Module):
    #   an implementation of Spatial Pyramid Pooling
    def __init__(self, spp_dim, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.level = spp_dim
        self.pool_type = pool_type
        for value, i in enumerate(spp_dim):
            key = 'pool_{}'.format(i)
            if self.pool_type == 'max_pool':
                self.register_parameter(key, nn.AdaptiveMaxPool2d([value,value]))
            elif self.pool_type == 'average_pool':
                self.register_parameter(key, nn.AdaptiveAvgPool2d([value, value]))
            else:
                raise ValueError(" wrong type error, should be max_pool or average_pool")

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.level):
            assert h<self.level[i] and w<self.level[i], "h and w is smaller than pool size"
            key = 'pool_{}'.format(i)
            tensor = self.parameters[key](x)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class DenseAffineGridGen(nn.Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = Variable(torch.from_numpy(self.grid.astype(np.float32)).cuda())


    def forward(self, input1):

        # self.batchgrid = self.grid.repeat(input1.size(0),1,1,1)  # batch channel height width
        # self.batchgrid = Variable(self.batchgrid).cuda()
        # auto boardcasting  need to check
        x = torch.add(self.batchgrid, input1)
        return x
