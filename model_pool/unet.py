import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from glob import glob

from .unet_expr import UNet3D
from .unet_expr2 import UNet3D2
from .unet_expr3 import UNet3D3
from .unet_expr4 import UNet3D4
from .unet_expr5 import UNet3D5
from .unet_expr4_test import UNet3Dt1
from .unet_expr4_test2 import UNet3Dt2
from .unet_expr4_test3 import UNet3Dt3
from .unet_expr4_test4 import UNet3Dt4
from .unet_expr4_test5 import UNet3Dt5
from .unet_expr4_test6 import UNet3Dt6
from .unet_expr4_test7 import UNet3Dt7
from .unet_expr4_test8 import UNet3Dt8
from .unet_expr4_test9 import UNet3Dt9
from .unet_expr_bon import UNet3DB
from .unet_expr_bon_s import UNet3DBS
from .unet_expr4_bon import UNet3D4B
from .unet_expr4_ens_nr import UNet3D4BNR
from .unet_expr5_bon import UNet3D5B
from .unet_expr5_ens import UNet3D5BE
from .unet_expr6_bon import UNet3D5BM
from .unet_expr7_bon import UNet3DB7
from .unet_expr8_bon import UNet3DB8
from .unet_expr9_bon import UNet3DB9
from .unet_expr10_bon import UNet3DB10
from .unet_expr11_bon import UNet3DB11
from .unet_expr12_bon import UNet3DB12
from .unet_expr13_bon import UNet3DB13
from .unet_expr14_bon import UNet3DB14
from .unet_expr15_bon import UNet3DB15
from .unet_expr16_bon import UNet3DB16
from .unet_expr17_bon import UNet3DB17
from .vnet_expr import VNet
from  .zhenlin_net import *
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

class Unet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        network_name =opt['tsk_set']['network_name']
        from .base_model import get_from_model_pool
        self.network = get_from_model_pool(network_name, self.n_in_channel, self.n_class)
        #self.network = CascadedModel([UNet_light1(n_in_channel,self.n_class,bias=True,BN=True)]+[UNet_light1(n_in_channel+self.n_class,self.n_class,bias=True,BN=True) for _ in range(3)],end2end=True, auto_context=True,residual=True)
        #self.network.apply(unet_weights_init)
        self.opt_optim =opt['tsk_set']['optim']
        self.init_optimize_instance(warmming_up=True)

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


    def init_optimize_instance(self, warmming_up=False):
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,
                                                                                   warmming_up=warmming_up)

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt_optim['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" no warming up the learning rate is {}".format(lr))


    def set_input(self, input, is_train=True):
        self. is_train = is_train
        if is_train:
            if not self.add_resampled:
                self.input = Variable(input[0]['image']).cuda()
            else:
                self.input =Variable(torch.cat((input[0]['image'], input[0]['resampled_img']),1)).cuda()

        else:
            self.input = Variable(input[0]['image'],volatile=True).cuda()
            if 'resampled_img' in input[0]:
                self.resam = Variable( input[0]['resampled_img']).cuda().volatile
        self.gt = Variable(input[0]['label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input):
        # here input should be Tensor, not Variable
        return self.network.forward(input)






    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        output = self.forward(self.input)
        if isinstance(output, list):
            self.output = output[-1]
            self.loss = self.cal_seq_loss(output)
        else:
            self.output = output
            self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()

