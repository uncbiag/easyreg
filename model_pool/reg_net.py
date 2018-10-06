import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from .reg_net_expr import *
from . import networks
from .losses import Loss
from .metrics import get_multi_metric
from data_pre.partition import Partition
#from model_pool.utils import weights_init
from model_pool.utils import *
from model_pool.mermaid_net import MermaidNet
import torch.nn as nn
import matplotlib.pyplot as plt
from model_pool.nn_interpolation import get_nn_interpolation
import SimpleITK as sitk

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
        which_epoch = opt['tsk_set']['which_epoch']
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        self.spacing = np.asarray(opt['tsk_set']['extra_info']['spacing'])

        input_img_sz = [int(self.img_sz[i]*self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        network_name =opt['tsk_set']['network_name']
        self.mermaid_on = True if 'mermaid' in network_name else False
        self.debug_sym_on = True if 'sym' in network_name else False

        self.network = model_pool[network_name](input_img_sz, opt)#AffineNetCycle(input_img_sz)#
        #self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        # if self.continue_train:
        #     self.load_network(self.network,'reg_net',which_epoch)
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.single_mod = opt['tsk_set']['single_mod']
        self.init_optimize_instance(warmming_up=True)
        self.step_count =0.
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')

    def init_optimize_instance(self, warmming_up=False):
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,self.network,
                                                                                   warmming_up=warmming_up)


    def adjust_learning_rate(self, new_lr=-1):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if new_lr<0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" no warming up the learning rate is {}".format(lr))


    def set_input(self, data, is_train=True):
        data[0]['image'] =data[0]['image'].cuda()
        data[0]['label'] =data[0]['label'].cuda()
        moving, target, l_moving,l_target = get_pair(data[0])
        input = data[0]['image']
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.input = input
        self.fname_list = list(data[1])


    def forward(self,input=None):
        # here input should be Tensor, not Variable
        if input is None:
            input =self.input
        return self.network.forward(input, self.moving,self.target)

    def phi_regularization(self):
        if len(self.disp.shape)>2:
            constr_map  = self.network.hessianField(self.disp)
            #constr_map = self.network.jacobiField(self.disp)
        else:
            constr_map = self.network.affine_cons(self.disp, sched='l2')

        reg = constr_map.sum()

        return reg


    def cal_loss(self):
        # output should be BxCx....
        # target should be Bx1x
        factor = 10# 1e-7
        factor = sigmoid_decay(self.cur_epoch,static=5, k=4)*factor
        sim_loss  = self.loss_fn.get_loss(self.output,self.target)
        reg_loss = self.phi_regularization() if self.disp is not None else 0.
        if self.iter_count%10==0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(), reg_loss.item(),factor))
        return sim_loss+reg_loss*factor

    def cal_mermaid_loss(self):
        loss_overall_energy, sim_energy, reg_energy = self.network.do_criterion_cal( self.moving,self.target, cur_epoch=self.cur_epoch)
        return loss_overall_energy  #*20 ###############################
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



    # get image paths
    def get_image_paths(self):
        return self.fname_list

    def backward_net(self):
        self.loss.backward()



    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.output, self.phi, self.disp = self.forward()
        if self.mermaid_on:
            self.loss = self.cal_mermaid_loss()
        elif self.debug_sym_on:
            self.loss = self.cal_sym_loss()
        else:
            self.loss = self.cal_loss()
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
        if self.single_mod:
            self.output, self.phi, self.disp= self.forward()
            self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi)
            warped_label_map_np= self.warped_label_map.detach().cpu().numpy()
            self.l_target_np= self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        else:
            step = 3
            print("Attention!!, the multi-step mode is on, {} step would be performed".format(step))
            phi = None
            for i in range(step):
                self.output, phi_cur, self.disp = self.forward()
                self.input = torch.cat((self.output,self.target),1)
                if i>0:
                    bilinear = Bilinear(zero_boundary=False)
                    phi_cur = bilinear(phi,phi_cur)
                phi = phi_cur
            self.phi = phi
            self.warped_label_map = self.get_warped_label_map(self.l_moving, self.phi)
            warped_label_map_np  =self.warped_label_map.detach().cpu().numpy()
            self.l_target_np = self.l_target.detach().cpu().numpy()
            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np, rm_bg=False)
        # if not self.print_val_detail:
        #     print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))
        # else:
        #     print('batch_avg_res{}'.format(self.val_res_dic['batch_avg_res']))
        #     print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))





    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)



    def save_fig_3D(self,phase):
        saving_folder_path = os.path.join(self.record_path, '3D')
        make_dir(saving_folder_path)
        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_"+phase+ "_iter_" + str(self.iter_count)
            saving_file_path = saving_folder_path + '/' + appendix + "_moving.nii.gz"
            output = sitk.GetImageFromArray(self.moving[i, 0, ...])
            output.SetSpacing(self.spacing)
            sitk.WriteImage(output, saving_file_path)
            saving_file_path = saving_folder_path + '/' + appendix + "_target.nii.gz"
            output = sitk.GetImageFromArray(self.target[i, 0, ...])
            output.SetSpacing(self.spacing)
            sitk.WriteImage(output, saving_file_path)
            saving_file_path = saving_folder_path + '/' + appendix + "_reproduce.nii.gz"
            output = sitk.GetImageFromArray(self.output[i, 0, ...])
            output.SetSpacing(self.spacing)
            sitk.WriteImage(output, saving_file_path)

    def save_fig_2D(self,phase):
        saving_folder_path = os.path.join(self.record_path, '2D')
        make_dir(saving_folder_path)

        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_"+phase+"_iter_" + str(self.iter_count)
            save_image_with_scale(saving_folder_path + '/' + appendix + "_moving.tif", self.moving[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_target.tif", self.target[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_reproduce.tif", self.output[i, 0, ...])

    def save_fig(self,phase,standard_record=False,saving_gt=True):
        pass
        from model_pool.visualize_registration_results import  show_current_images
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
        if self.disp is not None and len(self.disp.shape)>2 and not self.mermaid_on:
            disp = ((self.disp[:,...]**2).sum(1))**0.5


        if self.mermaid_on:
            disp = self.disp[:,0,...]
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



