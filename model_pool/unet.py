import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from .unet_expr import UNet3D
from . import networks
from .losses import Loss
from .metrics import get_multi_metric
import torch.optim.lr_scheduler as lr_scheduler
from data_pre.partition import Partition



class Unet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = 1
        self.n_class = opt['tsk_set']['extra_info']['num_label']
        which_epoch = opt['tsk_set']['which_epoch']
        self.img_sz = opt['tsk_set']['extra_info']['img_sz']
        continue_train = opt['tsk_set']['continue_train']
        tile_sz =  opt['dataset']['tile_size']
        overlap_size = opt['dataset']['overlap_size']
        padding_mode = opt['dataset']['padding_mode']
        self.iter_count = 0
        self.network = UNet3D(n_in_channel,self.n_class)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        if continue_train:
            self.load_network(self.network,'unet',which_epoch)
        self.loss_fn = Loss(opt)
        self.init_optim(opt['tsk_set']['optim'])
        self.partition = Partition(tile_sz, overlap_size, padding_mode)
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')

    def init_optim(self,opt):
        optimize_name = opt['optim_type']
        lr = opt['lr']
        beta = opt['adam']['beta']
        lr_sched_opt = opt['lr_scheduler']
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name =='adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas= (beta, 0.999))
        else:
            self.optimzer =  torch.optim.SGD(self.network.parameters(),lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler=None
        self.exp_lr_scheduler = None
        if self.lr_sched_type=='custom':
            step_size = lr_sched_opt['custom']['step_size']
            gamma = lr_sched_opt['custom']['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            self.exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, factor=factor, verbose=True,
                                                       threshold=threshold, min_lr=min_lr)




    def set_input(self, input, is_train=True):
        if is_train:
            self.input = Variable(input['image']).cuda()
        else:
            self.input = Variable(input['image'],volatile=True).cuda()
        self.gt = Variable(input['label']).long().cuda()
        self.fname = input['file_id']


    def forward(self,input=None):
        # here input should be Tensor, not Variable
        if input is None:
            input =self.input
        return self.network.forward(input)


    def cal_loss(self):

        return self.loss_fn.get_loss(self.output,self.gt)


    # get image paths
    def get_image_paths(self):
        return self.fname

    def backward_net(self):
        self.loss.backward()


    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.output = self.forward()
        self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss.data[0]

    def get_assamble_pred(self,split_size=6):
        output = []
        self.input = torch.unsqueeze(torch.squeeze(self.input),1)
        input_split = torch.split(self.input, split_size=split_size)
        for input in input_split:
            output.append(self.forward(input).cpu())
        pred_patched =  torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data,1)[1]
        self.output = self.partition.assemble(pred_patched, self.img_sz)
        print('debugging')

    def get_evaluation(self):
        gt= self.gt.data.cpu().numpy()
        self.val_res_dic = get_multi_metric(np.expand_dims(self.output,0),gt,rm_bg=False)
        print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))


    def cal_val_errors(self):
        self.cal_test_errors()
        if self.exp_lr_scheduler is not None:
            self.exp_lr_scheduler.step(self.loss.data[0])

    def cal_test_errors(self):
        self.get_assamble_pred()
        self.get_evaluation()

    def get_val_res(self):
        return self.val_res_dic['batch_label_avg_res']['dice']


    def get_current_visuals(self):
        return OrderedDict([('input', self.input), ('output', self.output)])

    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)





