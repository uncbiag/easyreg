import numpy as np
import torch
import os
from time import time

from .base_mermaid import MermaidBase

from .network_pool import *
from .net_utils import print_network
from .losses import Loss
from .metrics import get_multi_metric
from model_pool.utils import *
from model_pool.mermaid_net import MermaidNet
from model_pool.voxel_morph import VoxelMorphCVPR2018,VoxelMorphMICCAI2019

model_pool = {'affine_sim':AffineNet,
              'affine_unet':Affine_unet,
              'affine_cycle':AffineNetCycle,
              'affine_sym': AffineNetSym,
              'mermaid':MermaidNet,
              'vm_cvpr':VoxelMorphCVPR2018,
              'vm_miccai':VoxelMorphMICCAI2019
              }




class RegNet(MermaidBase):

    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        MermaidBase.initialize(self,opt)
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        input_img_sz = opt['dataset']['img_after_resize']
        self.input_img_sz = input_img_sz
        self.spacing = np.asarray(opt['dataset']['spacing_to_refer']) if self.use_physical_coord else 1. / (np.array(input_img_sz) - 1)
        self.spacing = normalize_spacing(self.spacing,self.input_img_sz) if self.use_physical_coord else self.spacing
        network_name =opt['tsk_set']['network_name']
        self.affine_on = True if 'affine' in network_name else False
        self.nonp_on = True if 'mermaid' in network_name else False
        self.using_affine_sym = True if self.affine_on and 'sym' in network_name else False
        self.network = model_pool[network_name](input_img_sz, opt) #AffineNetCycle(input_img_sz)#
        #self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.init_optimize_instance(warmming_up=True)
        self.step_count =0.
        print('---------- Networks initialized -------------')
        print_network(self.network)
        print('-----------------------------------------------')


    def init_optimize_instance(self, warmming_up=False):
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,self.network,
                                                                                   warmming_up=warmming_up)


    def adjust_learning_rate(self, new_lr=-1):
        if new_lr<0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" the learning rate now is set to {}".format(lr))


    def set_input(self, data, is_train=True):
        img_and_label, self.fname_list = data
        self.pair_path = data[0]['pair_path']
        img_and_label['image'] =img_and_label['image'].cuda()
        if 'label' in img_and_label:
            img_and_label['label'] =img_and_label['label'].cuda()
        moving, target, l_moving,l_target = get_pair(img_and_label)
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.original_im_sz = data[0]['original_sz']
        self.original_spacing = data[0]['original_spacing']








    def cal_affine_loss(self,output=None,disp_or_afparam=None,using_decay_factor=False):
        factor = 1.0
        if using_decay_factor:
            factor = sigmoid_decay(self.cur_epoch,static=5, k=4)*factor
        if self.loss_fn.criterion is not None:
            sim_loss  = self.loss_fn.get_loss(output,self.target)
        else:
            sim_loss = self.network.get_sim_loss(output,self.target)
        reg_loss = self.network.scale_reg_loss(disp_or_afparam) if disp_or_afparam is not None else 0.
        if self.iter_count%10==0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(), reg_loss.item(),factor))
        return sim_loss+reg_loss*factor



    def cal_affine_sym_loss(self):
        sim_loss = self.network.sym_sim_loss(self.loss_fn.get_loss,self.moving,self.target)
        sym_reg_loss = self.network.sym_reg_loss(bias_factor=1.)
        scale_reg_loss = self.network.scale_reg_loss(sched = 'l2')
        factor_scale = 10 #1e-3  # 1  ############################# TODo #####################
        factor_scale = float(max(sigmoid_decay(self.cur_epoch, static=30, k=3) * factor_scale,0.1))  #################static 1 TODO ##################3
        factor_scale = float( max(1e-3,factor_scale))
        factor_sym =10#10
        sim_factor = 1
        loss = sim_factor*sim_loss + factor_sym * sym_reg_loss + factor_scale * scale_reg_loss
        if self.iter_count%10==0:
            print('sim_loss:{}, factor_sym: {}, sym_reg_loss: {}, factor_scale {}, scale_reg_loss: {}'.format(
                sim_loss.item(),factor_sym,sym_reg_loss.item(),factor_scale,scale_reg_loss.item())
            )

        return loss

    def cal_mermaid_loss(self):
        loss = self.network.get_loss()
        return loss


    def backward_net(self, loss):
        loss.backward()

    def get_debug_info(self):
        info = {'file_name':self.fname_list}
        return info

    def forward(self, input=None):
        if hasattr(self.network, 'set_cur_epoch'):
            self.network.set_cur_epoch(self.cur_epoch)
        output, phi, disp_or_afparam= self.network.forward(self.moving, self.target)
        if self.nonp_on:
            loss = self.cal_mermaid_loss()
        elif self.using_affine_sym:
            loss = self.cal_affine_sym_loss()
        else:
            loss=self.cal_affine_loss(output,disp_or_afparam,using_decay_factor=self.affine_on)
        return output, phi, disp_or_afparam, loss

    def optimize_parameters(self,input=None):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.output, self.phi, self.disp_or_afparam,loss = self.forward()

        self.backward_net(loss)
        self.loss = loss.item()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss






    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_evaluation()

    def get_evaluation(self):
        s1 = time()
        self.output, self.phi, self.disp_or_afparam,_= self.forward()
        self.warped_label_map=None
        if self.l_moving is not None:
            self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi,use_01=False)
            print("!!!!!!!!!!!!!!!!testing the time cost is {}".format(time() - s1))
            warped_label_map_np= self.warped_label_map.detach().cpu().numpy()
            self.l_target_np= self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        self.jacobi_val = self.compute_jacobi_map((self.phi).detach().cpu().numpy(), crop_boundary=True, use_01=False)
        print("current batch jacobi is {}".format(self.jacobi_val))

    def get_extra_res(self):
        return self.jacobi_val


    def save_image_into_original_sz_with_given_reference(self):
        inverse_phi = self.network.get_inverse_map(use_01=False)
        self._save_image_into_original_sz_with_given_reference(self.pair_path, self.original_im_sz[0], self.phi, inverse_phi=inverse_phi, use_01=False)


    def get_extra_to_plot(self):
        return self.network.get_extra_to_plot()



    def save_deformation(self):
        if not self.affine_on:
            import nibabel as nib
            phi_np = self.phi.detach().cpu().numpy()
            phi_np = (phi_np+1.)/2.  # normalize the phi into 0, 1
            for i in range(phi_np.shape[0]):
                phi = nib.Nifti1Image(phi_np[i], np.eye(4))
                nib.save(phi, os.path.join(self.record_path, self.fname_list[i]) + '_phi.nii.gz')
        else:
            # todo the affine param is assumed in -1, 1 phi coord, to be fixed into 0,1 coord
            affine_param = self.disp_or_afparam
            if isinstance(affine_param,list):
                affine_param = self.disp_or_afparam[0]
            affine_param = affine_param.detach().cpu().numpy()
            for i in range(affine_param.shape[0]):
                np.save( os.path.join(self.record_path, self.fname_list[i]) + 'affine_param.npy',affine_param[i])


    def set_train(self):
        self.network.train(True)
        self.is_train =True
        torch.set_grad_enabled(True)


    def set_val(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_debug(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_test(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)



