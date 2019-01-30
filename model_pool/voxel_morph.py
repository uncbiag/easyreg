#!/usr/bin/env python
"""
registration network described in voxelmorph
Created by zhenlinx on 11/8/18
"""

import os
import sys

sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))

from model_pool.net_utils import gen_identity_map
import mermaid.pyreg.finite_differences as fdt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functions.bilinear import *

class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2)):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        self.residual = residual

    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear(x)
        if self.residual:
            x += x

        return x


class VoxelMorphCVPR2018(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(VoxelMorphCVPR2018, self).__init__()
        enc_filters = [16, 32, 32, 32, 32]
        #dec_filters = [32, 32, 32, 8, 8]
        dec_filters = [32, 32, 32, 32,16,16]
        input_channel =2
        output_channel= 3
        self.input_channel = 2
        self.output_channel = 3
        self.img_sz = img_sz
        self.id_transform = gen_identity_map(self.img_sz, 1.0)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True)
        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        for i in range(len(dec_filters)):
            if i==0:
                self.decoders.append(convBlock(enc_filters[-1], dec_filters[i], stride=1, bias=True))
            elif i<4:
                self.decoders.append(convBlock(dec_filters[i-1] if i==4 else dec_filters[i - 1] + enc_filters[4-i],
                                            dec_filters[i], stride=1, bias=True))
            else:
                self.decoders.append(convBlock(dec_filters[i-1], dec_filters[i], stride=1, bias=True))

        self.flow = nn.Conv3d(dec_filters[-1] + enc_filters[0], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        # identity transform for computing displacement

    def forward(self, source, target):


        x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)


        x_dec_1 = self.decoders[0](F.interpolate(x_enc_5, scale_factor=2,mode='trilinear'))
        #del x_enc_5
        x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_4), dim=1), scale_factor=2,mode='trilinear'))
        #del x_dec_1, x_enc_4
        x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), scale_factor=2,mode='trilinear'))
        #del x_dec_2, x_enc_3
        x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        #del x_dec_3, x_enc_2
        x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, scale_factor=2,mode='trilinear'))
        #del x_dec_4
        x_dec_6 = self.decoders[5](x_dec_5)
        disp_field = self.flow(torch.cat((x_dec_6, x_enc_1), dim=1))
        #del x_dec_5, x_enc_1

        deform_field = disp_field + self.id_transform
        warped_source = self.bilinear(source, deform_field)
        return warped_source, deform_field, disp_field

    def scale_reg_loss(self,disp=None,sched='l2'):
        fd = fdt.FD_torch(np.array([1.,1.,1.]))
        dfx = fd.dXc(disp[:, 0, ...])
        dfy = fd.dYc(disp[:, 1, ...])
        dfz = fd.dZc(disp[:, 2, ...])
        l2 = dfx**2+dfy**2+dfz**2
        reg = l2.mean()
        return reg

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

def test():
    cuda = torch.device('cuda:0')

    # unet = UNet_light2(2,3).to(cuda)
    net = VoxelMorphCVPR2018([80, 192, 192]).to(cuda)
    print(net)
    with torch.enable_grad():
        input1 = torch.randn(1, 1, 80, 192, 192).to(cuda)
        input2 = torch.randn(1, 1, 80, 192, 192).to(cuda)
        disp_field, warped_input1, deform_field = net(input1, input2)
    pass

if __name__ == '__main__':
    test()