from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from model_pool.net_utils import *
#from mermaid.pyreg.forward_models import RHSLibrary





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

    def __init__(self):
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


        self.fc_1 = FcRel(4 * 3 * 6 * 6, 32, active_unit='relu')
        #self.fc_1 = FcRel(4 * 3 * 3 * 4, 32, active_unit='relu')
        self.fc_2 = FcRel(32, 12, active_unit='None')

    def forward(self, m,t):
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
        output = None
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
            raise('low resolution only')

        return output


class MomentumGen_resid(nn.Module):
    def __init__(self, low_res_factor=1, bn=False):
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
            raise('low resolution only')

        return output






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














class DisplaceAdd(nn.Module):
    """
    given displacement field,  add displacement on grid field
    """
    def __init__(self, img_sz,resize_factor):
        super(DisplaceAdd, self).__init__()
        self.height =img_sz[0]
        self.width =img_sz[1]
        img_sz = [int(img_sz[i]*resize_factor[i]) for i in range(dim)]
        self.grid = identity_map(img_sz)




    def forward(self, input1):

        # self.batchgrid = self.grid.repeat(input1.size(0),1,1,1)  # batch channel height width
        # self.batchgrid = Variable(self.batchgrid).cuda()
        # auto boardcasting  need to check
        x = torch.add(self.grid, input1)
        return x


class DisplaceAffine(nn.Module):
    """
    given displacement field,  add displacement on grid field
    """
    def __init__(self, img_sz,resize_factor):
        super(DisplaceAffine, self).__init__()
        self.height =img_sz[0]
        self.width =img_sz[1]
        img_sz = [int(img_sz[i]*resize_factor[i]) for i in range(dim)]
        self.grid = identity_map(img_sz)
        self.bilinear = Bilinear




    def forward(self, input1):
        x = torch.add(self.grid, input1)
        return x







class MomConv(nn.Module):
    def __init__(self, bn=False):
        super(MomConv,self).__init__()
        self.encoder = nn.Sequential(
            conv_bn_rel(1, 8, kernel_size=3, stride=1, active_unit='elu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(8, 16,  kernel_size=3, stride=1, active_unit='elu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(16, 2,  kernel_size=3, stride=1, active_unit=None, same_padding=True, bn=bn, reverse=False))
        self.decoder = nn.Sequential(
            conv_bn_rel(4, 16, kernel_size=3, stride=1, active_unit='elu', same_padding=True, bn=bn, reverse=True),
            conv_bn_rel(16, 16, kernel_size=3, stride=1, active_unit=None, same_padding=True, bn=bn, reverse=True),
            conv_bn_rel(16, 1, kernel_size=3, stride=1, active_unit=None, same_padding=True, bn=bn, reverse=True))
    def forward(self, input1, input2):
        x1 = self.encoder(input1)
        x2 = self.encoder(input2)
        x = torch.cat((x1,x2),dim=1)
        x = self.decoder(x)
        return x


class FlowRNN(nn.Module):
    def __init__(self, img_sz, spacing, bn=False):
        super(FlowRNN, self).__init__()

        #low_linker(self, coder_output, phi)
        self.low_conv1 = nn.Sequential(
            conv_bn_rel(1, 16, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(16, 8, kernel_size=4, stride=2, active_unit=None, same_padding=True, bn=bn, reverse=True))
        self.low_conv2 = nn.Sequential(
            conv_bn_rel(2, 16, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(16, 8, kernel_size=4, stride=2, active_unit=None, same_padding=True, bn=bn, reverse=True))
        self.low_linker = nn.Sequential(
            conv_bn_rel(16, 8, kernel_size=3, stride=1, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(8, 1, kernel_size=3, stride=1, active_unit=None, same_padding=True, bn=bn, reverse=False))

        #coder(self, lam, phi, bn= True)
        self.mid_down_conv = nn.Sequential(
            conv_bn_rel(16, 64, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(64, 8,  kernel_size=4, stride=2, active_unit=None, same_padding=True, bn=bn, reverse=False))
        self.mid_up_conv = nn.Sequential(
            conv_bn_rel(8, 16, kernel_size=4, stride=2, active_unit='relu', same_padding=True, bn=bn, reverse=True),
            conv_bn_rel(16, 2, kernel_size=4, stride=2, active_unit=None, same_padding=True, bn=bn, reverse=True))
        #high_linker(self, vec_prev, coder_output)
        self.high_linker = nn.Sequential(
            conv_bn_rel(4, 16, kernel_size=3, stride=1, active_unit='relu', same_padding=True, bn=bn, reverse=False),
            conv_bn_rel(16, 2, kernel_size=3, stride=1, active_unit=None, same_padding=True, bn=bn, reverse=False))

        self.gird = DenseAffineGridGen(img_sz)
        self.advect_trans = RHSLibrary(spacing)

    def _xpyts(self, x, y, v):
        # x plus y times scalar
        return x+y*v

    def _xts(self, x, v):
        # x times scalar
        return x*v

    def _xpy(self, x, y):
        return x+y

    def forward(self, input, scale_mom, n_time):
        vec_size = scale_mom.size()
        init_vec = self.initVec(vec_size)    # init vec
        x_prev = init_vec
        x_s_next = scale_mom     # init scale_mom
        x = None # record displacement
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
            do_prev1 = self.advect_trans.rhs_advect_map_multiNC(o_prev,x)
            do_prev2 = self.advect_trans.rhs_advect_map_multiNC(self._xpyts(o_prev, do_prev1, 0.5),x)
            do_prev3 = self.advect_trans.rhs_advect_map_multiNC(self._xpyts(o_prev, do_prev2, 0.5),x)
            do_prev4 = self.advect_trans.rhs_advect_map_multiNC(self._xpy(o_prev, do_prev3),x)
            do_prev = do_prev1 / 6. + do_prev2 / 3. + do_prev3 / 3. + do_prev4 / 6.
            o_prev = o_prev + do_prev
            del do_prev

            #o_prev = torch.tanh(o_prev)
        return o_prev, x

    def initVec(self, size):
        return torch.cuda.FloatTensor(size[0], size[1]*2, size[2],size[3]).zero_()