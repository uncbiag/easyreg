from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mermaid.utils as py_utils
import torch

from .net_utils import *





class Affine_unet(nn.Module):

    def __init__(self):
        super(Affine_unet,self).__init__()
        #(W−F+2P)/S+1, W - input size, F - filter size, P - padding size, S - stride.
        # self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1,active_unit='relu', same_padding=True, bn=False)
        # self.down_path_2 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_4 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_8 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_16 = conv_bn_rel(32, 16, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.fc_1 = FcRel(16*5*12*12,144,active_unit='relu')
        # self.fc_2 = FcRel(144,12,active_unit = 'None')

        self.down_path_1 = conv_bn_rel(1, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2 = MaxPool(2,2)
        self.down_path_4 = conv_bn_rel(32, 16, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8 = MaxPool(2,2)
        self.down_path_16 = conv_bn_rel(16, 4, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_32 = MaxPool(2,2)

        self.fc_1 = FcRel(4 * 2 * 6 * 6, 32, active_unit='relu')
        self.fc_2 = FcRel(32, 12, active_unit='None')

    def forward(self, m,t):
        d1_m = self.down_path_1(m)
        d1_t = self.down_path_1(t)
        d1 = torch.cat((d1_m,d1_t),1)
        d2 = self.down_path_2(d1)
        d4 = self.down_path_4(d2)
        d8 = self.down_path_8(d4)
        d16 = self.down_path_16(d8)
        d32 = self.down_path_32(d16)
        fc1 = self.fc_1(d32.view(d32.shape[0],-1))
        fc2 = self.fc_2(fc1).view((d32.shape[0],-1))
        return fc2




class Affine_unet_im(nn.Module):

    def __init__(self, use_identity=False, fc_size=4*6*6*5):
        super(Affine_unet_im,self).__init__()

        self.down_path_1 = conv_bn_rel(1, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)

        self.down_path_2_1 = MaxPool(2,2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_1 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_1_t_4 = nn.Sequential(self.down_path_2_1,self.down_path_2_2,self.down_path_4_1,self.down_path_4_2)
        self.down_path_8_1 = MaxPool(2,2)
        self.down_path_8_2  = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_16_1 = conv_bn_rel(32, 16, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_16_2 = conv_bn_rel(16, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_32   = conv_bn_rel(16, 4, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_t_32 = nn.Sequential(self.down_path_8_1,self.down_path_8_2,self.down_path_16_1,self.down_path_16_2,
                                              self.down_path_32)

        # fc_size = 4*6*6*5    # oai  4*3*6*6  #lung 4*5*5*5  oasis 4*4*4*4 # brats 4*3*3*3 Z

        self.fc_1 = FcRel(fc_size, 32, active_unit='relu')
        self.fc_2 = FcRel(32, 12, active_unit='None')
        self.identityMap = None
    def forward(self, m,t):

        # if self.identityMap is None:
        #     self.identityMap = torch.zeros(12).cuda()
        #     self.identityMap[0] = 1.
        #     self.identityMap[4] = 1.
        #     self.identityMap[8] = 1.
        #
        #
        # return torch.cat([self.identityMap.unsqueeze(0)]*m.shape[0], dim=0)


        d1_m = self.down_path_1(m)
        d1_t = self.down_path_1(t)
        d1 = torch.cat((d1_m,d1_t),1)
        d4 = self.down_path_1_t_4(d1)
        d32 = self.down_path_4_t_32(d4)
        fc1 = self.fc_1(d32.view(d32.shape[0],-1))
        fc2 = self.fc_2(fc1).view((d32.shape[0],-1))
        return fc2




class MomentumGen(nn.Module):
    def __init__(self, low_res_factor=1):
        super(MomentumGen,self).__init__()
        self.low_res_factor = low_res_factor
        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False,group=2)
        self.down_path_2 = conv_bn_rel(16, 32, 3, stride=2, active_unit='leaky_relu', same_padding=True, bn=False,group=2)
        self.down_path_4 = conv_bn_rel(32, 32, 3, stride=2, active_unit='leaky_relu', same_padding=True, bn=False)
        self.down_path_8 = conv_bn_rel(32, 32, 3, stride=2, active_unit='leaky_relu', same_padding=True, bn=False)
        self.down_path_16 = conv_bn_rel(32, 32, 3, stride=2, active_unit='leaky_relu', same_padding=True, bn=False)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,
                                     reverse=True)
        self.up_path_4 = conv_bn_rel(64, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,
                                     reverse=True)
        self.up_path_2_1 = conv_bn_rel(64, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,
                                       reverse=True)
        if low_res_factor==1  or low_res_factor==None or low_res_factor ==[1.,1.,1.]:
            self.up_path_2_2 = conv_bn_rel(64, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
            self.up_path_1_1 = conv_bn_rel(8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=False, reverse=True)
            self.up_path_1_2 = conv_bn_rel(24, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)
        elif low_res_factor ==0.5 :
            self.up_path_2_2 = conv_bn_rel(32, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)

    def forward(self, x):
        output = None
        d1 = self.down_path_1(x)
        d2 = self.down_path_2(d1)
        d4 = self.down_path_4(d2)
        d8 = self.down_path_8(d4)
        d16 = self.down_path_16(d8)
        u8 = self.up_path_8(d16)
        u4 = self.up_path_4(torch.cat((u8, d8), 1))
        del d8
        u2_1 = self.up_path_2_1(torch.cat((u4, d4), 1))
        del d4
        if self.low_res_factor==1:
            u2_2 = self.up_path_2_2(torch.cat((u2_1, d2), 1))
            del d2
            u1_1 = self.up_path_1_1(u2_2)
            output = self.up_path_1_2(torch.cat((u1_1, d1), 1))
            del d1
        elif self.low_res_factor==0.5:
            output = self.up_path_2_2(u2_1)

        return output



class MomentumGen_im(nn.Module):
    def __init__(self, low_res_factor=1,bn=False):
        super(MomentumGen_im,self).__init__()
        self.low_res_factor = low_res_factor
        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_1 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_4_1 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16 = conv_bn_rel(64, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_8_2= conv_bn_rel(128, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_1 = conv_bn_rel(64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_4_2 = conv_bn_rel(96, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        if low_res_factor==1  or low_res_factor==None or low_res_factor ==[1.,1.,1.]:
            self.up_path_2_2 = conv_bn_rel(64, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
            self.up_path_1_1 = conv_bn_rel(8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=bn, reverse=True)
            self.up_path_1_2 = conv_bn_rel(24, 3, 3, stride=1, active_unit='None', same_padding=True, bn=bn)
        elif low_res_factor ==0.5 :
            self.up_path_2_2 = conv_bn_rel(64, 16, 3, stride=1, active_unit='None', same_padding=True)
            self.up_path_2_3 = conv_bn_rel(16, 3, 3, stride=1, active_unit='None', same_padding=True)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d4_1 = self.down_path_4_1(d2_2)
        d4_2 = self.down_path_4_2(d4_1)
        d8_1 = self.down_path_8_1(d4_2)
        d8_2 = self.down_path_8_2(d8_1)
        d16 = self.down_path_16(d8_2)


        u8_1 = self.up_path_8_1(d16)
        u8_2 = self.up_path_8_2(torch.cat((d8_2,u8_1),1))
        u4_1 = self.up_path_4_1(u8_2)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1))
        u2_1 = self.up_path_2_1(u4_2)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1))
        output = self.up_path_2_3(u2_2)
        if not self.low_res_factor==0.5:
            raise('for now. only half sz downsampling is supported')

        return output


class MomentumGen_resid(nn.Module):
    def __init__(self, low_res_factor=1, bn=False, adaptive_mode=False):
        super(MomentumGen_resid,self).__init__()
        self.low_res_factor = low_res_factor
        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_1 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(64, 128, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_2 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_1 = conv_bn_rel(128, 256, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(256, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_8_2= conv_bn_rel(128+128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_8_3= conv_bn_rel(128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_1 = conv_bn_rel(128, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_4_2 = conv_bn_rel(64+64, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_2_2 = conv_bn_rel(32+32, 16, 3, stride=1, active_unit='None', same_padding=True)
        self.up_path_2_3 = conv_bn_rel(16, 3, 3, stride=1, active_unit='None', same_padding=True)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2+ d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2


        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3,u8_1),1))
        u8_3 = self.up_path_8_3(u8_2)
        u8_3 = u8_2 + u8_3
        u4_1 = self.up_path_4_1(u8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3,u4_1),1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        output = self.up_path_2_3(u2_2)
        if not self.low_res_factor==0.5:
            raise('for now. only half sz downsampling is supported')

        return output



class Seg_resid(nn.Module):
    def __init__(self, num_class, bn=False):
        super(Seg_resid,self).__init__()
        self.down_path_1 = conv_bn_rel(1, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_1 = conv_bn_rel(64, 128, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_2 = conv_bn_rel(128,128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(128, 256, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_3 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)

        self.up_path_4_1 = conv_bn_rel(256, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_4_2 = conv_bn_rel(128+128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(128, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_2_2 = conv_bn_rel(128+64, 96, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_3 = conv_bn_rel(96, 96, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_1_1 = conv_bn_rel(96, 96, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_1_2 = conv_bn_rel(96+32, 64, 3, stride=1, active_unit='leaky_relu',same_padding=True, bn=bn)
        self.up_path_1_3 = conv_bn_rel(64, num_class, 3, stride=1, active_unit='leaky_relu', same_padding=True)


    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2+ d8_3
        u4_1 = self.up_path_4_1(d8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3,u4_1),1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        u2_3 = self.up_path_2_3(u2_2)
        u1_1 = self.up_path_1_1(u2_3)
        u1_2 = self.up_path_1_2(torch.cat((d1, u1_1), 1))
        u1_3 = self.up_path_1_3(u1_2)
        output = u1_3

        return output



class Seg_resid_imp(nn.Module):
    def __init__(self, num_class, bn=False):
        super(Seg_resid_imp,self).__init__()
        self.down_path_1 = conv_bn_rel(1, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_1 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(64, 128, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_2 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_1 = conv_bn_rel(128, 256, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)

        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(256, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_8_2 = conv_bn_rel(128+128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_1 = conv_bn_rel(128, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_4_2 = conv_bn_rel(64+64, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_2_2 = conv_bn_rel(64+32, 48, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_3 = conv_bn_rel(48, 48, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_1_1 = conv_bn_rel(48, 48, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_1_2 = conv_bn_rel(48+32, 64, 3, stride=1, active_unit='leaky_relu',same_padding=True, bn=bn)
        self.up_path_1_3 = conv_bn_rel(64, num_class, 3, stride=1, active_unit='leaky_relu', same_padding=True)


    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2+ d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2


        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3,u8_1),1))
        u8_3 = self.up_path_8_3(u8_2)
        u8_3 = u8_2 + u8_3
        u4_1 = self.up_path_4_1(u8_3)
        u4_1 = self.up_path_4_1(d8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3,u4_1),1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        u2_3 = self.up_path_2_3(u2_2)
        u1_1 = self.up_path_1_1(u2_3)
        u1_2 = self.up_path_1_2(torch.cat((d1, u1_1), 1))
        u1_3 = self.up_path_1_3(u1_2)
        output = u1_3

        return output
