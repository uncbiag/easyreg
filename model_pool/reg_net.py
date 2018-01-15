import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from .reg_net_expr import SimpleNet
from . import networks
from .losses import Loss
from .metrics import get_multi_metric
from data_pre.partition import Partition
from model_pool.utils import weights_init
from model_pool.utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
from model_pool.nn_interpolation import get_nn_interpolation
import SimpleITK as sitk



class RegNet(BaseModel):


    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        which_epoch = opt['tsk_set']['which_epoch']
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        self.network = SimpleNet(self.img_sz)
        #self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        if self.continue_train:
            self.load_network(self.network,'reg_net',which_epoch)
        self.loss_fn = Loss(opt)
        self.init_optim(opt['tsk_set']['optim'])
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')




    def set_input(self, data, is_train=True):
        moving, target, l_moving,l_target = get_pair(data[0])
        input = organize_data(moving, target, sched='depth_concat')
        volatile = not is_train
        self.moving = Variable(moving.cuda(),volatile=volatile)
        self.target = Variable(target.cuda(),volatile=volatile)
        self.l_moving = Variable(l_moving.cuda(),volatile=volatile)
        self.l_target = Variable(l_target.cuda(),volatile=volatile)
        self.input = Variable(input.cuda(),volatile=volatile)
        self.fname_list = list(data[1])


    def forward(self,input=None):
        # here input should be Tensor, not Variable
        if input is None:
            input =self.input
        return self.network.forward(input, self.moving)


    def cal_loss(self):
        # output should be BxCx....
        # target should be Bx1x
        return self.loss_fn.get_loss(self.output,self.target)


    # get image paths
    def get_image_paths(self):
        return self.fname_list

    def backward_net(self):
        self.loss.backward()


    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.output, self.phi = self.forward()
        self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss.data[0]

    def get_warped_label_map(self,label_map, phi, sched='nn'):
        if sched == 'nn':
            warped_label_map = get_nn_interpolation(label_map, phi)
            # check if here should be add assert
            assert abs(torch.sum(
                warped_label_map.data - warped_label_map.data.round())) < 0.1, "nn interpolation is not precise"
        else:
            raise ValueError(" the label warpping method is not implemented")
        return warped_label_map

    def cal_val_errors(self):
        self.cal_test_errors()
        if self.exp_lr_scheduler is not None:
            self.exp_lr_scheduler.step(self.get_val_res())

    def cal_test_errors(self):
        self.get_evaluation()

    def get_evaluation(self):
        self.output, self.phi = self.forward()
        warped_label_map_np = self.get_warped_label_map(self.l_moving,self.phi).data.cpu().numpy()
        self.l_target_np= self.l_target.data.cpu().numpy()

        self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        if not self.print_val_detail:
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))
        else:
            print('batch_avg_res{}'.format(self.val_res_dic['batch_avg_res']))
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))


    def get_val_res(self):
        return self.val_res_dic['batch_label_avg_res']['dice']




    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)

    def save_fig(self):
        if self.dim == 2:
            self.save_fig_2D()
        else:
            self.save_fig_3D()

    def save_fig_3D(self):
        saving_folder_path = os.path.join(self.record_path, '3D')
        make_dir(saving_folder_path)
        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_iter_" + str(self.iter_count)
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

    def save_fig_2D(self):
        saving_folder_path = os.path.join(self.record_path, '2D')
        make_dir(saving_folder_path)

        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_iter_" + str(self.iter_count)
            save_image_with_scale(saving_folder_path + '/' + appendix + "_moving.tif", self.moving[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_target.tif", self.target[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_reproduce.tif", self.output[i, 0, ...])





