import numpy as np
import torch
import os
from time import time

from .base_model import BaseModel
from .network_pool import *
from .net_utils import print_network
from .losses import Loss
from .metrics import get_multi_metric
from model_pool.utils import *
from model_pool.mermaid_net import MermaidNet
from model_pool.nn_interpolation import get_nn_interpolation

model_pool = {'affine_sim':AffineNet,
              'affine_unet':Affine_unet,
              'affine_cycle':AffineNetCycle,
              'affine_sym': AffineNetSym,
              'mermaid':MermaidNet
              }




class RegNet(BaseModel):


    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        self.spacing = np.asarray(opt['tsk_set']['extra_info']['spacing'])

        input_img_sz = [int(self.img_sz[i]*self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        network_name =opt['tsk_set']['network_name']
        self.mermaid_on = True if 'mermaid' in network_name else False
        self.using_sym_loss = True if 'sym' in network_name else False

        self.network = model_pool[network_name](input_img_sz, opt)#AffineNetCycle(input_img_sz)#
        #self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.single_mod = opt['tsk_set']['reg'][('single_mod',True,'whether iter the whole model')]
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
        img_and_label['image'] =img_and_label['image'].cuda()
        if 'label' in img_and_label:
            img_and_label['label'] =img_and_label['label'].cuda()
        moving, target, l_moving,l_target = get_pair(img_and_label)
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target


    def forward(self,input=None):
        if hasattr(self.network, 'set_cur_epoch'):
            self.network.set_cur_epoch(self.cur_epoch)
        return self.network.forward(self.moving,self.target)



    def cal_loss(self):
        # output should be BxCx....
        # target should be Bx1x
        factor = 10# 1e-7
        factor = sigmoid_decay(self.cur_epoch,static=5, k=4)*factor
        sim_loss  = self.loss_fn.get_loss(self.output,self.target)
        reg_loss = self.network.scale_reg_loss() if self.extra is not None else 0.
        if self.iter_count%10==0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(), reg_loss.item(),factor))
        return sim_loss+reg_loss*factor

    def cal_mermaid_loss(self):
        loss_overall_energy, sim_energy, reg_energy = self.network.do_criterion_cal( self.moving,self.target, cur_epoch=self.cur_epoch)
        return loss_overall_energy
    def cal_sym_loss(self):
        sim_loss = self.network.sym_sim_loss(self.loss_fn.get_loss,self.moving,self.target)
        sym_reg_loss = self.network.sym_reg_loss(bias_factor=1.)
        scale_reg_loss = self.network.scale_reg_loss(sched = 'l2')
        factor_scale =1  # 1  ############################# TODo #####################
        factor_scale = float(max(sigmoid_decay(self.cur_epoch, static=30, k=3) * factor_scale,0.1))  #################static 1 TODO ##################3
        factor_scale = float( max(1e-3,factor_scale))
        factor_sym =1#10 ################################### ToDo ####################################
        sim_factor = 1

        loss = sim_factor*sim_loss + factor_sym * sym_reg_loss + factor_scale * scale_reg_loss

        if self.iter_count%10==0:
            print('sim_loss:{}, factor_sym: {}, sym_reg_loss: {}, factor_scale {}, scale_reg_loss: {}'.format(
                sim_loss.item(),factor_sym,sym_reg_loss.item(),factor_scale,scale_reg_loss.item())
            )

        return loss




    def backward_net(self):
        self.loss.backward()



    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.output, self.phi, self.extra = self.forward()
        if self.mermaid_on:
            self.loss = self.cal_mermaid_loss()
        elif self.using_sym_loss:
            self.loss = self.cal_sym_loss()
        else:
            self.loss=self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def update_loss(self,epoch, end_of_epoch):
        pass

    def get_current_errors(self):
        return self.loss.item()


    def get_warped_img_map(self,img, phi):
        bilinear = Bilinear()
        warped_img_map = bilinear(img, phi)

        return warped_img_map


    def get_warped_label_map(self,label_map, phi, sched='nn'):
        if sched == 'nn':
            warped_label_map = get_nn_interpolation(label_map, phi)
            # check if here should be add assert
            assert abs(torch.sum(
                warped_label_map.detach() - warped_label_map.detach().round())) < 0.1, "nn interpolation is not precise"
        else:
            raise ValueError(" the label warpping method is not implemented")
        return warped_label_map

    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_evaluation()

    def get_evaluation(self):

        self.output, self.phi, self.extra= self.forward()
        self.warped_label_map=None
        if self.l_moving is not None:
            self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi)
            warped_label_map_np= self.warped_label_map.detach().cpu().numpy()
            self.l_target_np= self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        self.jacobi_val = self.compute_jacobi_map((self.phi).detach().cpu().numpy() )
        print("current batch jacobi is {}".format(self.jacobi_val))

    def get_extra_res(self):
        return self.jacobi_val




    def save_fig(self,phase,standard_record=False,saving_gt=True):
        from model_pool.visualize_registration_results import show_current_images
        visual_param={}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = self.record_path
        visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 2
        visual_param['pair_path'] = self.fname_list
        visual_param['iter'] = phase+"_iter_" + str(self.iter_count)
        disp=None
        extra_title = 'disp'
        if self.extra is not None and len(self.extra.shape)>2 and not self.mermaid_on:
            disp = ((self.extra[:,...]**2).sum(1))**0.5


        if self.mermaid_on:
            disp = self.extra[:,0,...]
            extra_title='affine'
        show_current_images(self.iter_count,  self.moving, self.target,self.output, self.l_moving,self.l_target,self.warped_label_map,
                            disp, extra_title, self.phi, visual_param=visual_param)

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



