import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from .unet_expr import UNet3D
from . import networks
from .losses import init_loss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_pre.partition import Partition



class Unet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = 1
        n_class = opt.n_class
        lr= opt['tsk_set']['optim']['lr']
        beta= opt['tsk_set']['optim']['adam']['beta']
        which_epoch = opt['tsk']['which_epoch']
        optimize_name = opt['tsk']['optim']['optimizer']
        tile_sz =  opt['dataset']['tile_size']
        overlap_size = opt['dataset']['overlap_size']
        padding_mode = opt['dataset']['padding_mode']
        self.iter_count = 0
        self.network = UNet3D(n_in_channel,n_class)
        self.criticUpdates = 1
        if opt.continue_train:
            self.load_network(self.network,'unet',which_epoch)
        if optimize_name =='adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas= (beta, 0.999))
        else:
            self.optimzer =  torch.optim.SGD(self.network.parameters(),lr=lr)
        self.optimizer.zero_grad()
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.loss_fn = init_loss(opt)
        self.partition = Partition(tile_sz, overlap_size, padding_mode)
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')


    def set_input(self, input, is_train=True):
        if is_train:
            self.input = Variable(input['img'])
        else:
            self.input = Variable(input['img'],volatile=True)

        self.img_sz =input['img'].size()
        self.gt = Variable(input['label'])
        self.fname = input['file_id']


    def forward(self):
        # here input should be Tensor, not Variable
        return self.network.forward(self.input)


    def cal_loss(self):
        return self.loss_fn.get_loss(self.output,self.gt)




    # get image paths
    def get_image_paths(self):
        return self.fname

    def backward_net(self):
        self.loss.backward()


    def optimize_parameters(self):
        self.iter_count+=1
        self.exp_lr_scheduler.step()
        self.output = self.forward()
        self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss.data[0]

    def get_assamble_pred(self,split_size=2):
        output = []
        input_split = torch.split(self.input, split_size=split_size)
        for input in input_split:
            self.set_input(input)
            output.append(self.network())
        pred_patched =  torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data,1)[1]
        output = self.partition.assemble(pred_patched)
        self.output = output[:,:,self.img_sz[2],self.img_sz[3],self.img_sz[4]]




    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_assamble_pred()
        self.cal_loss()


    def get_current_visuals(self):
        return OrderedDict([('input', self.input), ('output', self.output)])

    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)





