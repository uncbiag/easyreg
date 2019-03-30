import numpy as np
import torch
import os
from time import time

from .base_model import BaseModel
from .network_pool import *
from .net_utils import print_network
from .losses import Loss
from .metrics import get_multi_metric
import mermaid.pyreg.finite_differences as fdt
from model_pool.utils import *
from model_pool.mermaid_net_mutli_channel import MermaidNet
from model_pool.voxel_morph import VoxelMorphCVPR2018,VoxelMorphMICCAI2019
try:
    from model_pool.nn_interpolation import get_nn_interpolation
except:
    pass
from mermaid.pyreg.utils import compute_warped_image_multiNC
model_pool = {'affine_sim':AffineNet,
              'affine_unet':Affine_unet,
              'affine_cycle':AffineNetCycle,
              'affine_sym': AffineNetSym,
              'mermaid':MermaidNet,
              'vm_cvpr':VoxelMorphCVPR2018,
              'vm_miccai':VoxelMorphMICCAI2019
              }




class RegNet(BaseModel):


    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        self.print_val_detail = opt['tsk_set']['print_val_detail']

        input_img_sz = [int(self.img_sz[i]*self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        self.spacing = np.asarray(opt['tsk_set'][('spacing',1. / (np.array(input_img_sz) - 1),'spacing')])

        network_name =opt['tsk_set']['network_name']
        self.mermaid_on = True if 'mermaid' in network_name else False
        self.using_sym_loss = True if 'sym' in network_name else False
        self.using_affine = True if 'affine' in network_name else False

        self.network = model_pool[network_name](input_img_sz, opt)#AffineNetCycle(input_img_sz)#
        #self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.single_mod = opt['tsk_set']['reg'][('single_mod',True,'whether iter the whole model')]
        self.init_optimize_instance(warmming_up=True)
        self.step_count =0.
        self.multi_gpu_on = False
        print('---------- Networks initialized -------------')
        print_network(self.network)
        print('-----------------------------------------------')


    def init_optimize_instance(self, warmming_up=False):
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,self.network,
                                                                                   warmming_up=warmming_up)


    def adjust_learning_rate(self, new_lr=-1):
        if new_lr<0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" the learning rate now is set to {}".format(lr))


    def set_input(self, data, is_train=True):
        img_and_label, self.fname_list = data
        img_and_label['image'] =img_and_label['image'].cuda()
        if 'label' in img_and_label:
            img_and_label['label'] =img_and_label['label'].cuda()
        moving, target, l_moving,l_target = get_pair(img_and_label)
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target







    def cal_loss(self,using_decay_factor=False):
        # output should be BxCx....
        # target should be Bx1x
        from model_pool.global_variable import reg_factor_in_regnet
        factor = reg_factor_in_regnet# 1e-7
        if using_decay_factor:
            factor = sigmoid_decay(self.cur_epoch,static=5, k=4)*factor
        if self.loss_fn.criterion is not None:
            sim_loss  = self.loss_fn.get_loss(self.output,self.target)
        else:
            sim_loss = self.network.get_sim_loss(self.output,self.target)
        reg_loss = self.network.scale_reg_loss(self.disp_or_afparam) if self.disp_or_afparam is not None else 0.
        if self.iter_count%10==0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(), reg_loss.item(),factor))
        return sim_loss+reg_loss*factor

    def cal_mermaid_loss(self):
        loss_overall_energy = self.network.get_loss()
        return loss_overall_energy
    def cal_sym_loss(self):
        sim_loss = self.network.sym_sim_loss(self.loss_fn.get_loss,self.moving,self.target)
        sym_reg_loss = self.network.sym_reg_loss(bias_factor=1.)
        scale_reg_loss = self.network.scale_reg_loss(sched = 'l2')
        factor_scale =1e-3  # 1  ############################# TODo #####################
        # factor_scale = float(max(sigmoid_decay(self.cur_epoch, static=30, k=3) * factor_scale,0.1))  #################static 1 TODO ##################3
        # factor_scale = float( max(1e-3,factor_scale))
        factor_sym =10#10 ################################### ToDo ####################################
        sim_factor = 1

        loss = sim_factor*sim_loss + factor_sym * sym_reg_loss + factor_scale * scale_reg_loss

        if self.iter_count%10==0:
            print('sim_loss:{}, factor_sym: {}, sym_reg_loss: {}, factor_scale {}, scale_reg_loss: {}'.format(
                sim_loss.item(),factor_sym,sym_reg_loss.item(),factor_scale,scale_reg_loss.item())
            )

        return loss




    def backward_net(self, loss):
        loss.backward()

    def get_debug_info(self):
        info = {'file_name':self.fname_list}
        return info

    def forward(self, input=None):
        if hasattr(self.network, 'set_cur_epoch'):
            self.network.set_cur_epoch(self.cur_epoch)
        output, phi, disp_or_afparam= self.network.forward(self.moving, self.target)
        if self.mermaid_on:
            loss = self.cal_mermaid_loss()
        elif self.using_sym_loss:
            loss = self.cal_sym_loss()
        else:
            loss=self.cal_loss(using_decay_factor=self.using_affine)
        return output, phi, disp_or_afparam, loss

    def optimize_parameters(self,input=None):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.output, self.phi, self.disp_or_afparam,loss = self.forward()

        self.backward_net(loss)
        self.loss = loss.item()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()


    def get_current_errors(self):
        return self.loss


    def get_warped_img_map(self,img, phi):
        bilinear = Bilinear()
        warped_img_map = bilinear(img, phi)

        return warped_img_map


    def get_warped_label_map(self,label_map, phi, sched='nn'):
        if sched == 'nn':
            ###########TODO temporal comment for torch1 compatability
            try:
                print(" the cuda nn interpolation is used")
                warped_label_map = get_nn_interpolation(label_map, phi)
            except:
                warped_label_map = compute_warped_image_multiNC(label_map,phi,self.spacing,spline_order=0,zero_boundary=True,use_01_input=False)
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
        s1 = time()
        self.output, self.phi, self.disp_or_afparam,_= self.forward()
        self.warped_label_map=None
        if self.l_moving is not None:
            self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi)
            print("!!!!!!!!!!!!!!!!testing the time cost is {}".format(time() - s1))
            warped_label_map_np= self.warped_label_map.detach().cpu().numpy()
            self.l_target_np= self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        self.jacobi_val = self.compute_jacobi_map((self.phi).detach().cpu().numpy() )
        print("current batch jacobi is {}".format(self.jacobi_val))

    def get_extra_res(self):
        return self.jacobi_val

    def compute_jacobi_map(self,map):
        """ here we compute the jacobi in numpy coord. It is consistant to jacobi in image coord only when
          the image direction matrix is identity."""
        from model_pool.global_variable import save_jacobi_map
        import SimpleITK as sitk
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        spacing = 2. / (np.array(input_img_sz) - 1)  # the disp coorindate is [-1,1]
        fd = fdt.FD_np(spacing)
        dfx = fd.dXc(map[:, 0, ...])
        dfy = fd.dYc(map[:, 1, ...])
        dfz = fd.dZc(map[:, 2, ...])
        jacobi_det = dfx * dfy * dfz
        # self.temp_save_Jacobi_image(jacobi_det,map)
        jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])  #
        jacobi_num = np.sum(jacobi_det < 0.)
        print("print folds for each channel {},{},{}".format(np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        jacobi_abs_mean = jacobi_abs / map.shape[0]
        jacobi_num_mean = jacobi_num / map.shape[0]
        self.jacobi_map = None
        if save_jacobi_map:
            jacobi_abs_map = np.abs(jacobi_det)
            for i in range(jacobi_abs_map.shape[0]):
                jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
                pth = os.path.join(self.record_path, self.fname_list[i] + 'jacobi_img.nii')
                sitk.WriteImage(jacobi_img, pth)
            self.jacobi_map =jacobi_abs_map
        return jacobi_abs_mean, jacobi_num_mean




    def save_fig(self,phase,standard_record=False,saving_gt=True):
        from model_pool.visualize_registration_results import show_current_images
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
        extraImage, extraName = self.network.get_extra_to_plot()
        extra_title = 'disp'
        if self.disp_or_afparam is not None and len(self.disp_or_afparam.shape)>2 and not self.mermaid_on:
            disp = ((self.disp_or_afparam[:,...]**2).sum(1))**0.5


        if self.mermaid_on:
            disp = self.disp_or_afparam[:,0,...]
            extra_title='affine'
        if self.jacobi_map is not None:
            disp = self.jacobi_map
            extra_title = 'jacobi det'
        show_current_images(self.iter_count, iS=self.moving,iT=self.target,iW=self.output,
                            iSL=self.l_moving,iTL=self.l_target, iWL=self.warped_label_map,
                            vizImages=disp, vizName=extra_title,phiWarped=self.phi,
                            visual_param=visual_param,extraImages=extraImage, extraName= extraName)

    def save_deformation(self):
        if not self.using_affine:
            import nibabel as nib
            phi_np = self.phi.detach().cpu().numpy()
            for i in range(phi_np.shape[0]):
                phi = nib.Nifti1Image(phi_np[i], np.eye(4))
                nib.save(phi, os.path.join(self.record_path, self.fname_list[i]) + '_phi.nii.gz')
        else:
            affine_param = self.disp_or_afparam
            if isinstance(affine_param,list):
                affine_param = self.disp_or_afparam[0]
            affine_param = affine_param.detach().cpu().numpy()
            for i in range(affine_param.shape[0]):
                np.save( os.path.join(self.record_path, self.fname_list[i]) + 'affine_param.npy',affine_param[i])


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



