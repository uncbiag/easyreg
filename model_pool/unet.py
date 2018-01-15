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
from model_pool.utils import weights_init
from model_pool.utils import *
import torch.nn as nn
import matplotlib.pyplot as plt



class Unet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = 1
        self.n_class = opt['tsk_set']['extra_info']['num_label']
        which_epoch = opt['tsk_set']['which_epoch']
        self.print_val_detail = opt['tsk_set']['print_val_detail']

        tile_sz =  opt['dataset']['tile_size']
        overlap_size = opt['dataset']['overlap_size']
        padding_mode = opt['dataset']['padding_mode']
        self.network = UNet3D(n_in_channel,self.n_class)
        #self.network.apply(weights_init)
        if self.continue_train:
            self.load_network(self.network,'unet',which_epoch)
        self.loss_fn = Loss(opt)
        self.init_optim(opt['tsk_set']['optim'])
        self.partition = Partition(tile_sz, overlap_size, padding_mode)
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)
        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')



    def set_input(self, input, is_train=True):
        if is_train:
            self.input = Variable(input[0]['image']).cuda()
        else:
            self.input = Variable(input[0]['image'],volatile=True).cuda()
        self.gt = Variable(input[0]['label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input=None):
        # here input should be Tensor, not Variable
        if input is None:
            input =self.input
        return self.network.forward(input)


    def cal_loss(self):
        """"
        output should be B x n_class x ...
        gt    should be B x 1 x.......
        """
        return self.loss_fn.get_loss(self.output,self.gt)


    # get image paths
    def get_image_paths(self):
        return self.fname_list

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



    def get_evaluation(self):
        self.gt_np= self.gt.data.cpu().numpy()
        self.val_res_dic = get_multi_metric(np.expand_dims(self.output,0), self.gt_np,rm_bg=False)
        if not self.print_val_detail:
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))
        else:
            print('batch_avg_res{}'.format(self.val_res_dic['batch_avg_res']))
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))


    def save_val_fig(self,mode):
        show_current_images_3d(self.output,np.squeeze(self.gt_np))
        fig=None
        if True:
            fig = plt.gcf()
            folder_name = os.path.join(self.record_path, mode)
            make_dir(folder_name)
            file_name = self.fname_list[0] + '_iter_' + str(self.iter_count)
            fig.savefig(os.path.join(folder_name, file_name), dpi=500)
        if False:
            plt.show()
        fig.clf()
        plt.clf()

    def cal_val_errors(self):
        self.cal_test_errors()
        if self.exp_lr_scheduler is not None:
            self.exp_lr_scheduler.step(self.get_val_res())

    def cal_test_errors(self):
        self.get_assamble_pred()
        self.get_evaluation()

    def get_val_res(self):
        return self.val_res_dic['batch_label_avg_res']['dice']


    def get_current_visuals(self):
        return OrderedDict([('input', self.input), ('output', self.output)])

    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)





