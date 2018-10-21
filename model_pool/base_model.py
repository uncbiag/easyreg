import os
from collections import OrderedDict

import torch

from data_pre.partition import Partition
from model_pool.losses import Loss
from model_pool.metrics import get_multi_metric
from model_pool.utils import *
import torch.optim.lr_scheduler as lr_scheduler
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
from .unet_expr4_test10 import UNet3Dt9_sim
from .unet_expr_bon import UNet3DB
from .unet_expr_bon_loc import UNet3DB_loc
from .unet_expr_bon_s import UNet3DBS
from .unet_expr_bon_s_prelu import UNet3DBS_Prelu
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
from .vonet_pool import UNet_asm
from .vonet_pool_un import UNet_asm_full,Vonet_test
from .vonet_pool_t9 import UNet_asm_t9
from .vonet_pool_t9_concise import UNet_asm_t9_con
from .vonet_pool_sim_prelu import UNet_asm_sim_prelu
from .prior_net import  PriorNet
from .gb_net_pool import  gbNet
from .unet_expr_extreme_deep import UNet3D_Deep
from .unet_expr_multi_mod import UNet3DMM
import SimpleITK as sitk
from glob import glob
import mermaid.pyreg.finite_differences as fdt




model_pool_1 = {
    'UNet3D': UNet3D,
    'UNet3D2': UNet3D2,
    'UNet3D3': UNet3D3,
    'UNet3D4': UNet3D4,
    'UNet3D5': UNet3D5,
    'UNet3Dt1': UNet3Dt1,
    'UNet3Dt2': UNet3Dt2,
    'UNet3Dt3': UNet3Dt3,
    'UNet3Dt4': UNet3Dt4,
    'UNet3Dt5': UNet3Dt5,
    'UNet3Dt6': UNet3Dt6,
    'UNet3Dt7': UNet3Dt7,
    'UNet3Dt8': UNet3Dt8,
    'UNet3Dt9': UNet3Dt9,
    'UNet3Dt9_sim':UNet3Dt9_sim,
    'UNet3DBS_Prelu':UNet3DBS_Prelu,
    'UNet3DB': UNet3DB,
    'UNet3DB_loc':UNet3DB_loc,
    'UNet3DBS': UNet3DBS,
    'UNet3D4B': UNet3D4B,
    'UNet3D4BNR': UNet3D4BNR,
    'UNet3D5B': UNet3D5B,
    'UNet3D5BE': UNet3D5BE,
    'UNet3D5BM': UNet3D5BM,
    'UNet3DB7': UNet3DB7,
    'UNet3DB8': UNet3DB8,
    'UNet3DB9': UNet3DB9,
    'UNet3DB10': UNet3DB10,
    'UNet3DB11': UNet3DB11,
    'UNet3DB12': UNet3DB12,
    'UNet3DB13': UNet3DB13,
    'UNet3DB14': UNet3DB14,
    'UNet3DB15': UNet3DB15,
    'UNet3DB16': UNet3DB16,
    'UNet3DB17': UNet3DB17,
    'VNet': VNet,
    'UNet_asm':UNet_asm,
    'UNet_asm_f':UNet_asm_full,
    'UNet_asm_t9':UNet_asm_t9,
    'UNet_asm_t9_con':UNet_asm_t9_con,
    'UNet_asm_sim_prelu':UNet_asm_sim_prelu,
    'Vonet_test':Vonet_test,
    'UNet3D_Deep':UNet3D_Deep,
    'UNet3DMM':UNet3DMM,
    'prior_net':PriorNet,
    'gb_net':gbNet
}


def get_from_model_pool(model_name,n_in_channel, n_class):
        if model_name in model_pool_1:
            return model_pool_1[model_name](n_in_channel, n_class)
        if model_name =='Cascaded_light1_4':
            model = CascadedModel([UNet_light1(n_in_channel,n_class,bias=True,BN=True)]+[UNet_light1(n_in_channel+n_class,n_class,bias=True,BN=True) for _ in range(3)],end2end=True, auto_context=True,residual=True)
            return model


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt['tsk_set']['gpu_ids']
        self.isTrain = opt['tsk_set']['train']
        self.save_dir = opt['tsk_set']['path']['check_point_path']
        self.record_path = opt['tsk_set']['path']['record_path']
        self.img_sz = opt['tsk_set']['img_size']
        self.spacing = None
        self.continue_train = opt['tsk_set']['continue_train']
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.n_in_channel = opt['tsk_set']['n_in_channel']
        self.input_resize_factor = opt['tsk_set']['input_resize_factor']
        self.optimizer= None
        self.lr_scheduler = None
        self.exp_lr_scheduler= None
        self.iter_count = 0
        self.dim = len(self.img_sz)
        self.network =None
        self.val_res_dic = {}





    def set_input(self, input):
        self.input = input

    def forward(self,input):
        pass

    def test(self):
        pass

    def set_train(self):
        self.network.train(True)
        self.is_train =True
    def set_val(self):
        self.network.train(False)
        self.is_train = False

    def set_debug(self):
        self.network.train(False)
        self.is_train = False

    def set_test(self):
        self.network.train(False)
        self.is_train = False



    def optimize_parameters(self):
        pass

    def init_optim(self, opt,network, warmming_up = False):
        optimize_name = opt['optim_type']
        if not warmming_up:
            lr = opt['lr']
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = 1e-4
            print(" warming up on the learning rate is {}".format(lr))
        beta = opt['adam']['beta']
        lr_sched_opt = opt['lr_scheduler']
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name == 'adam':
            re_optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            re_optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        re_optimizer.zero_grad()
        re_lr_scheduler = None
        re_exp_lr_scheduler = None
        if self.lr_sched_type == 'custom':
            step_size = lr_sched_opt['custom']['step_size']
            gamma = lr_sched_opt['custom']['gamma']
            re_lr_scheduler = torch.optim.lr_scheduler.StepLR(re_optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            re_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(re_optimizer, mode='min', patience=patience,
                                                                   factor=factor, verbose=True,
                                                                   threshold=threshold, min_lr=min_lr)
        return re_optimizer,re_lr_scheduler,re_exp_lr_scheduler



    def save_2d_visualize(self,input,gt,output):
        image_summary = make_image_summary(input, gt, output)




    # get image paths
    def get_image_paths(self):
        return self.fname_list


    def set_cur_epoch(self,epoch):
        self.cur_epoch = epoch
        self.cur_epoch_beg_tag = True


    def backward_net(self):
        self.loss.backward()


    def cal_loss(self,output= None):
       pass

    def cal_seq_loss(self,output_seq):
        loss =0.0
        for output in output_seq:
            loss += self.cal_loss(output)
        return loss


    def get_current_errors(self):
            return self.loss.data[0]



    def compute_jacobi_map(self,map,norm_zero = True):
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        spacing = 1. / (np.array(input_img_sz) - 1)
        fd = fdt.FD_np(spacing)
        dfx= fd.dXc(map[:, 0, ...])
        dfy= fd.dYc(map[:, 1, ...])
        dfz= fd.dZc(map[:, 2, ...])
        if norm_zero:
            jacobi_abs = np.sum(dfx<0.) + np.sum(dfy<0.) + np.sum(dfz<0.)
        else:
            jacobi_abs = np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        jacobi_abs_mean = jacobi_abs/ map.shape[0] #/ np.prod(map.shape)
        return jacobi_abs_mean



    def cal_val_errors(self, split_size=2):
        self.cal_test_errors(split_size)

    def cal_test_errors(self,split_size=2):
       pass


    def update_loss(self, epoch, end_of_epoch):
        pass


    def get_val_res(self, detail = False):
        if not detail:
            return np.mean(self.val_res_dic['batch_avg_res']['dice'][0,1:]), self.val_res_dic['batch_avg_res']['dice']
        else:
            return np.mean(self.val_res_dic['batch_avg_res']['dice'][0,1:]), self.val_res_dic['multi_metric_res']


    def get_test_res(self, detail=False):
        return self.get_val_res(detail = detail)

    def get_extra_res(self):
        return None


    def save_fig(self,phase,standard_record=False,saving_gt=True):
       pass


    def check_and_update_model(self,epoch):
        return None


    def do_some_clean(self):
        self.loss= None
        self.gt=None
        self.input = None
        self.output= None











