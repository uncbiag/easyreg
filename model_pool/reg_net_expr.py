from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from model_pool.modules import *
from functions.bilinear import *
from models.net_utils import init_weights

class SimpleNet(nn.Module):
    def __init__(self, img_sz=None, resize_factor=1.):
        super(SimpleNet,self).__init__()
        self.img_sz = img_sz
        self.denseGen = DisGen_Simple()
        init_weights(self.denseGen,init_type='kaiming')
        self.hessianField = HessianField()
        self.jacobiField = JacobiField()
        self.identity_map= gen_identity_map(self.img_sz,resize_factor)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        disField = self.denseGen(input)
        #hessianField = self.hessianField(disField)
        gridField = torch.add(self.identity_map,disField)
        #gridField= torch.tanh(gridField)
        output = self.bilinear(moving,gridField)
        return output, gridField, disField

class AffineNet(nn.Module):
    """
    here we need two affine net,
    1. do the affine by training single forward network
    # in this case if we want to do the affine
    we need to  warp the image then made it as a new input

    2. do affine by training a cycle network
    in this case, it is like svf , we would feed raw image once, and the
    network would warp the phi, the advantage of this method is we don't need
    to warp the image for several time as interpolation would introduce unstability
    """
    def __init__(self, img_sz=None, resize_factor=1.):
        super(AffineNet, self).__init__()
        self.img_sz = img_sz if len(img_sz)<4 else img_sz[2:]
        self.dim = len(self.img_sz)
        self.affine_gen = Affine_unet_im()
        self.affine_cons= AffineConstrain()
        self.phi= gen_identity_map(self.img_sz)
        self.bilinear = Bilinear()

    def gen_affine_map(self,Ab):
        Ab = Ab.view( Ab.shape[0],4,3 ) # 3d: (batch,3)
        phi = self.phi.view(self.dim, -1)
        affine_map = None
        # if self.dim == 2:
        #     affine_map[0, ...] = Ab[0] * self.phi[0, ...] + Ab[2] * self.phi[1, ...] + Ab[4]  # a_11x+a_21y+b1
        #     affine_map[1, ...] = Ab[1] * self.phi[0, ...] + Ab[3] * self.phi[1, ...] + Ab[5]  # a_12x+a_22y+b2
        # elif self.dim == 3:
        #     affine_map[0, ...] = Ab[0] * self.phi[0, ...] + Ab[3] * self.phi[1, ...] + Ab[6] * self.phi[2, ...] + Ab[9]
        #     affine_map[1, ...] = Ab[1] * self.phi[0, ...] + Ab[4] * self.phi[1, ...] + Ab[7] * self.phi[2, ...] + Ab[10]
        #     affine_map[2, ...] = Ab[2] * self.phi[0, ...] + Ab[5] * self.phi[1, ...] + Ab[8] * self.phi[2, ...] + Ab[11]
        if self.dim == 3:
            affine_map = torch.matmul( Ab[:,:3,:], phi)
            affine_map = Ab[:,3,:].contiguous().view(-1,3,1) + affine_map
            affine_map= affine_map.view([Ab.shape[0]] + list(self.phi.shape))
        return affine_map


    def forward(self,input,moving,target=None):
        affine_param = self.affine_gen(moving,target)
        affine_map = self.gen_affine_map(affine_param)
        #affine_map=affine_map.repeat(input.shape[0],1,1,1,1)
        output = self.bilinear(moving,affine_map)
        return output, affine_map, affine_param




class AffineNetCycle(nn.Module):   # is not implemented, need to be done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    here we need two affine net,
    1. do the affine by training single forward network
    # in this case if we want to do the affine
    we need to  warp the image then made it as a new input

    2. do affine by training a cycle network
    in this case, it is like svf , we would feed raw image once, and the
    network would warp the phi, the advantage of this method is we don't need
    to warp the image for several time as interpolation would introduce unstability
    """
    def __init__(self, img_sz=None, resize_factor=1.):
        super(AffineNetCycle, self).__init__()
        self.img_sz = img_sz
        self.dim = len(img_sz)
        self.step = 3
        self.affine_gen = Affine_unet()
        self.affine_cons= AffineConstrain()
        self.phi= gen_identity_map(self.img_sz)
        self.bilinear =Bilinear()



    def gen_affine_map(self,Ab):
        Ab = Ab.view( Ab.shape[0],4,3 ) # 3d: (batch,3)
        phi = self.phi.view(self.dim, -1)
        affine_map = None
        if self.dim == 3:
            affine_map = torch.matmul( Ab[:,:3,:], phi)
            affine_map = Ab[:,3,:].contiguous().view(-1,3,1) + affine_map
            affine_map= affine_map.view([Ab.shape[0]] + list(self.phi.shape))
        return affine_map

    def update_affine_param(self, cur_af, last_af): # A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2
        cur_af = cur_af.view(cur_af.shape[0], 4, 3)
        last_af = last_af.view(last_af.shape[0],4,3)
        updated_af = Variable(torch.zeros_like(cur_af.data)).cuda()
        if self.dim==3:
            updated_af[:,:3,:] = torch.matmul(cur_af[:,:3,:],last_af[:,:3,:])
            updated_af[:,3,:] = cur_af[:,3,:] + torch.squeeze(torch.matmul(cur_af[:,:3,:], torch.transpose(last_af[:,3:,:],1,2)),2)
        updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
        return updated_af



    def forward(self,input,moving,target):
        output = None
        moving_cp = moving
        affine_param_last = None
        bilinear = [Bilinear() for i in range(self.step)]

        for i in range(self.step):
            affine_param = self.affine_gen(moving,target)
            if i >0:
                affine_param = self.update_affine_param(affine_param,affine_param_last)
            affine_param_last = affine_param
            affine_map = self.gen_affine_map(affine_param)
            output = bilinear[i](moving_cp,affine_map)
            moving = output

        return output, affine_map, affine_param



class AffineNetSym(nn.Module):   # is not implemented, need to be done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    here we need two affine net,
    1. do the affine by training single forward network
    # in this case if we want to do the affine
    we need to  warp the image then made it as a new input

    2. do affine by training a cycle network
    in this case, it is like svf , we would feed raw image once, and the
    network would warp the phi, the advantage of this method is we don't need
    to warp the image for several time as interpolation would introduce unstability
    """
    def __init__(self, img_sz=None, resize_factor=1.):
        super(AffineNetSym, self).__init__()
        self.img_sz = img_sz
        self.dim = len(img_sz)
        self.step = 5
        self.using_complex_net = False
        self.affine_gen = Affine_unet_im() if self.using_complex_net else Affine_unet()
        self.affine_cons= AffineConstrain()
        self.phi= gen_identity_map(self.img_sz)
        self.count =0
        self.gen_identity_ap()
        self.grid_sample = F.grid_sample
        self.using_cycle = True
        self.bilinear = Bilinear()



    def gen_affine_map(self,Ab):
        Ab = Ab.view( Ab.shape[0],4,3 ) # 3d: (batch,3)
        phi = self.phi.view(self.dim, -1)
        affine_map = None
        if self.dim == 3:
            affine_map = torch.matmul( Ab[:,:3,:], phi)
            affine_map = Ab[:,3,:].contiguous().view(-1,3,1) + affine_map
            affine_map= affine_map.view([Ab.shape[0]] + list(self.phi.shape))
        return affine_map


    def update_affine_param(self, cur_af, last_af): # A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2
        cur_af = cur_af.view(cur_af.shape[0], 4, 3)
        last_af = last_af.view(last_af.shape[0],4,3)
        updated_af = Variable(torch.zeros_like(cur_af.data)).cuda()
        if self.dim==3:
            updated_af[:,:3,:] = torch.matmul(cur_af[:,:3,:],last_af[:,:3,:])
            updated_af[:,3,:] = cur_af[:,3,:] + torch.squeeze(torch.matmul(cur_af[:,:3,:], torch.transpose(last_af[:,3:,:],1,2)),2)
        updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
        return updated_af

    def gen_identity_ap(self):
        self.affine_identity = Variable(torch.zeros(12)).cuda()
        self.affine_identity[0] = 1.
        self.affine_identity[4] = 1.
        self.affine_identity[8] = 1.

    def sym_reg_loss(self, bias_factor=1.):
        """
        y = ax+b = a(cy+d)+b = acy +ad+b =y
        then ac = I, ad+b = 0
        :return:
        """
        ap_st, ap_ts  = self.affine_param

        ap_st = ap_st.view(-1, 4, 3)
        ap_ts = ap_ts.view(-1, 4, 3)
        ac = None
        ad_b = None

        if self.dim == 3:
            ac = torch.matmul(ap_st[:, :3, :], ap_ts[:, :3, :])
            ad_b = - ap_st[:, 3, :] + torch.squeeze(
                torch.matmul(ap_st[:, :3, :], torch.transpose(ap_ts[:, 3:, :], 1, 2)), 2)
        identity_matrix = self.affine_identity.view(4,3)[:3,:3]

        linear_transfer_part = torch.sum((ac-identity_matrix)**2)
        translation_part = bias_factor * (torch.sum(ad_b**2))

        sym_reg_loss = linear_transfer_part + translation_part
        if self.count %10 ==0:
            print("linear_transfer_part:{}, translation_part:{}, bias_factor:{}".format(linear_transfer_part.cpu().data.numpy(), translation_part.cpu().data.numpy(),bias_factor))
        return sym_reg_loss/ap_st.shape[0]

    def sym_sim_loss(self,loss_fn,moving,target):
        output = self.output
        sim_st = loss_fn(output[0],target)
        sim_ts = loss_fn(output[1], moving)
        sim_loss = sim_st +sim_ts
        return sim_loss / moving.shape[0]

    def scale_reg_loss(self,sched='l2'):
        affine_param = self.affine_param
        if sched=='l2':
            loss = torch.sum((affine_param[0]-self.affine_identity)**2 + (affine_param[1]-self.affine_identity)**2 )\
                   / (affine_param[0].shape[0])
            return loss
        elif sched=='det':
            loss = 0.
            for j in range(2):
                for i in range(affine_param[j].shape[0]):
                    affine_matrix = affine_param[j][i,:9].contiguous().view(3,3)
                    loss += (torch.det(affine_matrix) -1.)**2
            return  loss / (affine_param[0].shape[0])


    def forward(self,input,moving, target):
        self.count += 1
        if  not self.using_cycle:
            return self.single_forward(input,moving,target)
        else:
            return self.cycle_forward(input, moving, target)



    def single_forward(self,input,moving,target):

        self.affine_param = None
        self.output = None
        bilinear = [Bilinear() for _ in range(2)]
        affine_param_st = self.affine_gen(moving,target)
        affine_param_ts = self.affine_gen(target,moving)
        affine_map_st = self.gen_affine_map(affine_param_st)
        affine_map_ts = self.gen_affine_map(affine_param_ts)
        output_st = bilinear[0](moving, affine_map_st)
        output_ts = bilinear[1](target, affine_map_ts)
        # output_st = self.grid_sample(moving,affine_map_st.permute([0,2,3,4,1]),mode='trilinear', padding_mode='border')
        # output_ts = self.grid_sample(target,affine_map_ts.permute([0,2,3,4,1]),mode='trilinear', padding_mode='border')
        output = (output_st, output_ts)
        affine_param = (affine_param_st, affine_param_ts)
        self.affine_param = affine_param
        self.output= output

        return output_st, affine_map_st, affine_param_st

    def cycle_forward(self,input,moving, target):
        moving_cp = moving
        target_cp = target
        affine_param_st_last = None
        affine_param_ts_last = None

        for i in range(self.step):
            bilinear = [Bilinear() for _ in range(2)]
            affine_param_st = self.affine_gen(moving, target_cp)
            affine_param_ts = self.affine_gen(target, moving_cp)
            if i > 0:
                affine_param_st = self.update_affine_param(affine_param_st, affine_param_st_last)
                affine_param_ts = self.update_affine_param(affine_param_ts, affine_param_ts_last)
            affine_param_st_last = affine_param_st
            affine_param_ts_last = affine_param_ts
            affine_map_st = self.gen_affine_map(affine_param_st)
            affine_map_ts = self.gen_affine_map(affine_param_ts)
            # output_st = self.grid_sample(moving_cp, affine_map_st.permute([0, 2, 3, 4, 1]), mode='trilinear',
            #                              padding_mode='border')
            # output_ts = self.grid_sample(target_cp, affine_map_ts.permute([0, 2, 3, 4, 1]), mode='trilinear',
            #                              padding_mode='border')
            output_st = bilinear[0](moving_cp, affine_map_st)
            output_ts = bilinear[1](target_cp, affine_map_ts)
            moving = output_st
            target = output_ts

        output = (output_st, output_ts)
        affine_param = (affine_param_st, affine_param_ts)
        self.affine_param = affine_param
        self.output = output

        return output_st, affine_map_st, affine_param_st




class MomentumNet(nn.Module):
    def __init__(self, low_res_factor):
        super(MomentumNet,self).__init__()
        self.low_res_factor = low_res_factor
        self.mom_gen = MomentumGen_im(low_res_factor)

    def forward(self,input):
        return self.mom_gen(input)





class FlowNet(nn.Module):
    def __init__(self, img_sz, spacing):
        super(FlowNet,self).__init__()
        self.img_sz = img_sz
        self.spacing = spacing
        self.momConv = MomConv(bn=False)
        self.jacobiField = JacobiField()
        self.flowRnn= FlowRNN(self.img_sz,self.spacing, bn=False)
        self.grid = grid_gen(img_sz)
        self.bilinear = Bilinear()
    def forward(self, input, moving):
        x = self.momConv(input[0], input[1])
        gridField, disField = self.flowRnn(self.grid,x, n_time=5)
        jacobDisField = self.jacobiField(disField)
        output = self.bilinear(moving,gridField)
        return output, gridField,disField
