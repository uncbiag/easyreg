from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import *






class ControlGen(nn.Module):
    """
    designed for generate warp points
    """
    def __init__(self, n_control, bn=False, use_spp=False, spp_levels=(8,16)):
        super(ControlGen, self).__init__()

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
    """
    a simple conv implementation, generate displacement field
    """
    def __init__(self, bn=False):
        super(DisGen, self).__init__()
        # Build a LSTM
        self.conv1 = nn.Sequential(ConvBnRel(2, 64, 3, active_unit='elu', same_padding=True, bn=bn))
        self.conv2 = nn.Sequential(ConvBnRel(64, 128, 3, active_unit='elu', same_padding=True, bn=bn),
                                   ConvBnRel(128, 128, 3, active_unit='elu', same_padding=True, bn=bn))
        self.conv3 = nn.Sequential(ConvBnRel(128, 2, 3, active_unit='elu', same_padding=True, bn=bn))



    def forward(self, im_data):
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)

        return x





class SPPLayer(nn.Module):
    #   an implementation of Spatial Pyramid Pooling
    """
    implementation of spatial pyrmaid pooling,
    """
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


def grid_gen(info):
    height, width = info['img_h'], info['img_w']
    grid = np.zeros([2, height, width], dtype=np.float32)
    grid[0, ...] = np.expand_dims(
        np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / height), 0), repeats=width, axis=0).T, 0)
    grid[1, ...] = np.expand_dims(
        np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / width), 0), repeats=height, axis=0), 0)
    # self.grid[:,:,2] = np.ones([self.height, self.width])
    return Variable(torch.from_numpy(grid.astype(np.float32)).cuda())


class DenseAffineGridGen(nn.Module):
    """
    given displacement field,  add displacement on grid field
    """
    def __init__(self, info):
        super(DenseAffineGridGen, self).__init__()
        self.height = info['img_h']
        self.width = info['img_w']
        self.grid = grid_gen(info)




    def forward(self, input1):

        # self.batchgrid = self.grid.repeat(input1.size(0),1,1,1)  # batch channel height width
        # self.batchgrid = Variable(self.batchgrid).cuda()
        # auto boardcasting  need to check
        x = torch.add(self.grid, input1)
        return x


class MomConv(nn.Module):
    def __init__(self, bn=False):
        super(MomConv,self).__init__()
        self.encoder = self.mid_down_conv = nn.Sequential(
            ConvBnRel(1, 32, kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(32, 64,  kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(64, 16,  kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=False))
        self.decoder = nn.Sequential(
            ConvBnRel(32, 64, kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=True),
            ConvBnRel(64, 64, kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=True),
            ConvBnRel(64, 1, kernel_size=4, stride=2, active_unit='elu', same_padding=True, bn=bn, reverse=True))
    def forward(self, input1, input2):
        x1 = self.encoder(input1)
        x2 = self.encoder(input2)
        x = torch.cat((x1,x2),dim=1)
        x = self.decoder(x)
        return x


class FlowRNN(nn.Module):
    def __init__(self, info, bn=False):
        super(FlowRNN, self).__init__()

        #low_linker(self, coder_output, phi)
        self.low_conv1 = nn.Sequential(
            ConvBnRel(1, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(64, 8, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True))
        self.low_conv2 = nn.Sequential(
            ConvBnRel(2, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(64, 8, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True))
        self.low_linker = nn.Sequential(
            ConvBnRel(16, 1, kernel_size=3, stride=1, active_unit='relu', same_padding=True, bn=bn, reverse=False))

        #coder(self, lam, phi, bn= True)
        self.mid_down_conv = nn.Sequential(
            ConvBnRel(16, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(64, 64,  kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(64, 8,  kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False))
        self.mid_up_conv = nn.Sequential(
            ConvBnRel(8, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True),
            ConvBnRel(64, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True),
            ConvBnRel(64, 2, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True))
        #high_linker(self, vec_prev, coder_output)
        self.high_linker = nn.Sequential(
            ConvBnRel(4, 32, kernel_size=3, stride=1, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            ConvBnRel(32, 2, kernel_size=3, stride=1, active_unit='relu', same_padding=True, bn=bn, reverse=False))

        self.gird = DenseAffineGridGen(info)



    def forward(self, input, scale_mom, n_time):
        vec_size = scale_mom.size()
        init_vec = self.initVec(vec_size)    # init vec
        x_prev = init_vec
        x_s_next = scale_mom     # init scale_mom
        o_prev = input.repeat(vec_size[0],1,1,1)   # init grid
        for i in range(n_time):
            x_s = self.low_conv1(x_s_next)
            x_g= self.low_conv2(o_prev)
            x = torch.cat((x_s,x_g),dim=1)
            x_s_next = self.low_linker(x)    # next scale_mom
            x = self.mid_down_conv(x)
            x = self.mid_up_conv(x)
            x_prev = torch.cat((x,x_prev), dim=1)
            x_prev = self.high_linker(x_prev)
            #o_prev = torch.tanh(x+ o_prev)
            o_prev = x + o_prev
            #o_prev = torch.tanh(o_prev)
        return o_prev, x

    def initVec(self, size):
        return Variable(torch.cuda.FloatTensor(size[0], size[1]*2, size[2],size[3]).zero_())