import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel

from  .zhenlin_net import *
from . import networks

from model_pool.utils import unet_weights_init,vnet_weights_init
from model_pool.utils import *


class Pnet(BaseModel):
    def name(self):
        return 'prior-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        network_name =opt['tsk_set']['network_name']
        from .base_model import get_from_model_pool
        from model_pool.losses import Loss
        num_class =2
        self.set_num_class(num_class)
        opt['tsk_set']['extra_info']['num_label'] = num_class

        self.loss_fn = Loss(opt)
        self.network = get_from_model_pool(network_name, self.n_in_channel, self.num_class)
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
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,self.network,
                                                                                   warmming_up=warmming_up)

    def save_fig(self,phase, standard_record=True,saving_gt=True):
        pass

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
        self.gt = Variable(input[0]['checked_label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input):
        # here input should be Tensor, not Variable
        return self.network.forward(input)


    def set_num_class(self, num_class):
        self.num_class = num_class



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





    def get_val_res(self):
        return self.val_res


    def get_evaluation(self,brats_eval_on=True):
        pass


    def get_assamble_pred(self,split_size=3, old_verison=False):
        output = []
        if old_verison:
            self.input = torch.unsqueeze(torch.squeeze(self.input),1)
        else:
            self.input = torch.squeeze(self.input,0)
        input_split = torch.split(self.input, split_size=split_size)
        volatile_status = input_split[0].volatile
        print("check the input_split volatile status :{}".format(volatile_status))
        for input in input_split:
            if self.add_resampled:
                resam = self.resam.expand(input.size(0),self.resam.size(1),self.resam.size(2),self.resam.size(3),self.resam.size(4))
                input = torch.cat((input,resam),1)
            if not volatile_status:
                input.volatile = True
            res = self.forward(input)
            if isinstance(res,list):
                res = res[-1]
            output.append(res.detach().cpu())
            del res
        pred_patched = torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data,1)[1]
        pred_res = np.sum(np.sum(pred_patched.cpu().numpy())>0)
        self.gt_np= self.gt.data.cpu().numpy()
        gt_res = np.sum(np.sum(self.gt_np)>0)
        self.val_res = np.sum(pred_res==(gt_res))

        print("pred/actual{}/{}".format(np.sum(pred_patched.cpu().numpy()),gt_res))
