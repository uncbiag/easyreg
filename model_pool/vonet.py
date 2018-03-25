import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from glob import glob
from  .zhenlin_net import *
from .vonet_pool import  Vonet_test
from . import networks
from .losses import Loss
from .metrics import get_multi_metric
import torch.optim.lr_scheduler as lr_scheduler
from data_pre.partition import Partition
from model_pool.utils import unet_weights_init,vnet_weights_init
from model_pool.utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import SimpleITK as sitk

class Vonet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = 4
        self.check_point_path = opt['tsk_set']['path']['check_point_path']


        self.start_saving_model =  opt['tsk_set']['voting']['start_saving_model']
        self.saving_voting_per_epoch =  opt['tsk_set']['voting']['saving_voting_per_epoch']
        self.voting_save_sched =opt['tsk_set']['voting']['voting_save_sched']
        network_name =opt['tsk_set']['network_name']

        #self.network = self.get_from_model_pool(network_name,n_in_channel, self.n_class)

        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        epoch_list = [i for i in range(240, 249,5)  ]  #range(245,249,2)] 79.34   (51,249,3) 79.66
        self.network = Vonet_test(n_in_channel,self.n_class, self.check_point_path, epoch_list, (old_gpu_id,cur_gpu_id), bias=True,BN=True)


        self.optimizer_fea, self.lr_scheduler_fea, self.exp_lr_scheduler_fea =self.init_optim(opt['tsk_set']['optim'])
        self.optimizer_dis, self.lr_scheduler_dis, self.exp_lr_scheduler_dis =self.init_optim(opt['tsk_set']['optim'])

        check_point_path = opt['tsk_set']['path']['check_point_path']
        self.asm_path = os.path.join(check_point_path,'asm_models')
        make_dir(self.asm_path)
        self.start_asm_learning = False
        # here we need to add training_eval_record which should contain several thing
        # first it should record the dice performance(the label it contained), and the avg (or weighted avg) dice inside
        # it may also has some function to put it into a prior queue, which based on patch performance
        # second it should record the times of being called, just for record, or we may put it as some standard, maybe change dataloader, shouldn't be familiar with pytorch source code
        # third it should contains the path of the file, in case we may use frequent sampling on low performance patch
        # forth it should record the labels in that patch and the label_density, which should be done during the data process
        #
        self.training_eval_record={}
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)

        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')





    def set_input(self, input, is_train=True):
        self. is_train = is_train
        if is_train and not self.start_asm_learning:
            self.input = Variable(input[0]['image']*2-1).cuda()
        else:
            self.input = Variable(input[0]['image']*2-1,volatile=True).cuda()
        self.gt = Variable(input[0]['label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input=None):
        # here input should be Tensor, not Variable
        if input is None:
            input =self.input
        if not self.start_asm_learning:
            output = self.network(input)
        else:
            output = self.network.net_fea.forward(input)
            for term in output:
                term.volatile = False
                term.requires_grad = False
                term.detach_()
            output = self.network.net_dis.forward(output)
        return output




    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        output = self.forward()
        if isinstance(output, list):
            self.output = output[-1]
            self.loss = self.cal_seq_loss(output)
        else:
            self.output = output
            self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer_dis.step()
            if not self.start_asm_learning:
                self.optimizer_fea.step()
                self.optimizer_fea.zero_grad()
            self.optimizer_dis.zero_grad()
        if self.auto_saving_model():
            torch.save(self.network.net_dis.state_dict(),self.asm_path+'/'+'epoch_'+str(self.cur_epoch))
            fea_path = self.asm_path+'/'+'fea'
            if not os.path.exists(fea_path):
                torch.save(self.network.net_fea.state_dict(),fea_path)
            self.start_asm_learning = True





    def auto_saving_model(self):
        #if self.voting_save_sched == 'default':
        log = self.cur_epoch > self.start_saving_model and self.cur_epoch % self.saving_voting_per_epoch==0
        log = log and self.cur_epoch_beg_tag
        self.cur_epoch_beg_tag = False
        return log


