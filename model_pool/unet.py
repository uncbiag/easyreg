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



class Unet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = opt.n_in_channel
        n_class = opt.n_class
        lr= opt.lr
        beta= opt.beta
        which_epoch = opt.which_epoch
        self.network = UNet3D(n_in_channel,n_class)
        ##############################################################
        self.criticUpdates = 1
        ##############################################################
        if opt.continue_train:
            self.load_network(self.network,'unet',which_epoch)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas= (beta, 0.999))
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.optimizer.zero_grad()
        self.loss_fn = init_loss(opt)
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')


    def set_input(self, input):
        self.input = input['img']
        self.gt = input['label']
        self.fname = input['file_id']


    def forward(self):
        # here input should be Tensor, not Variable
        input = Variable(self.input)
        self.output = self.network.forward(input)


    # get image paths
    def get_image_paths(self):
        return self.fname

    def backward_net(self):
        self.loss = self.loss_fn.get_loss(self.output,self.gt)

        self.loss.backward()


    def optimize_parameters(self):
        self.exp_lr_scheduler.step()
        self.forward()
        self.backward_net()
        for iter_d in range(self.criticUpdates):
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss.data[0]

    def get_current_visuals(self):
        return OrderedDict([('input', self.input), ('output', self.output)])

    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)





