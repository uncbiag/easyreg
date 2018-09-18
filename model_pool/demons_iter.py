import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable

from data_pre.reg_data_utils import get_file_name
from .base_model import BaseModel
from .reg_net_expr import *
from . import networks
from .losses import Loss
from .metrics import get_multi_metric
from data_pre.partition import Partition
#from model_pool.utils import weights_init
from model_pool.utils import *
#from model_pool.mermaid_net import MermaidNet
import torch.nn as nn
import matplotlib.pyplot as plt
from model_pool.nn_interpolation import get_nn_interpolation
import SimpleITK as sitk
from model_pool.demons_utils import *
import mermaid.pyreg.utils as py_utils

import mermaid.pyreg.simple_interface as SI
import mermaid.pyreg.fileio as FIO
class DemonsRegIter(BaseModel):
    import mermaid.pyreg.utils as py_utils
    def name(self):
        return 'demons_reg iter'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        which_epoch = opt['tsk_set']['which_epoch']
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        #self.spacing = np.asarray(opt['tsk_set']['extra_info']['spacing'])
        input_img_sz = [int(self.img_sz[i]*self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        self.spacing= 1. / (np.array(input_img_sz)-1)# np.array([0.00501306, 0.00261097, 0.00261097])*2
        self.resize_factor = opt['tsk_set']['input_resize_factor']

        network_name =opt['tsk_set']['network_name']
        self.network_name = network_name
        self.single_mod = True
        if network_name =='affine':
            self.affine_on = True
            self.demon_on = False
            raise ValueError("affine is not separately used in demons")

        elif network_name =='demons':
            self.affine_on = False
            self.demon_on = True
        self.si = SI.RegisterImagePair()
        self.im_io = FIO.ImageIO()
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.step_count =0.
        self.identity_map = py_utils.identity_map_multiN([1,1]+input_img_sz, self.spacing)*2-1
        self.identity_map = torch.from_numpy(self.identity_map).cuda()






    def set_input(self, data, is_train=True):
        data[0]['image'] =(data[0]['image'].cuda()+1)/2
        data[0]['label'] =data[0]['label'].cuda()
        moving, target, l_moving,l_target = get_pair(data[0])
        input = data[0]['image']
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.input = input
        self.fname_list = list(data[1])
        self.pair_path = data[0]['pair_path']
        self.pair_path = [path[0] for path in self.pair_path]
        self.resized_moving_path = self.resize_input_img_and_save_it_as_tmp(self.pair_path[0],is_label=False,fname='moving.nii.gz')
        self.resized_target_path = self.resize_input_img_and_save_it_as_tmp(self.pair_path[1],is_label= False, fname='target.nii.gz')
        self.resized_l_moving_path = self.resize_input_img_and_save_it_as_tmp(self.pair_path[2],is_label= True, fname='l_moving.nii.gz')
        self.resized_l_target_path = self.resize_input_img_and_save_it_as_tmp(self.pair_path[3],is_label= True, fname='l_target.nii.gz')


    def resize_input_img_and_save_it_as_tmp(self, img_pth, is_label=False,fname=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        img = self.__read_and_clean_itk_info(img_pth)
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud(self.resize_factor)
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        fpth = os.path.join(self.record_path,fname)
        #############################  be attention the original of the image is no consistent which may cause the demos fail
        # so this should be checked
        # img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        # img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        # img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth


    def __read_and_clean_itk_info(self,path):
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))





    def affine_optimization(self):

        output, loutput, phi = performDemonsRegistration(self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path)

        self.output = output
        self.warped_label_map = loutput

        self.disp = None
        self.phi = None
        # self.phi = self.phi*2-1

        return self.output, None, None


    def demons_optimization(self):
        output, loutput, phi = performDemonsRegistration(self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path)


        self.disp = None
        self.output = output
        self.warped_label_map = loutput

        self.phi = phi
        # self.phi = self.phi*2-1
        return self.output,None, None


    def forward(self,input=None):
        if self.affine_on and not self.demon_on:
            return self.affine_optimization()
        elif self.demon_on:
            #self.affine_optimization()
            return self.demons_optimization()






    # get image paths
    def get_image_paths(self):
        return self.fname_list



    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_evaluation()

    def get_evaluation(self):
        if self.single_mod:
            self.output, self.phi, self.disp= self.forward()
            #self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi)
            warped_label_map_np= self.warped_label_map
            self.l_target_np= self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        else:
            step = 8
            print("Attention!!, the multi-step mode is on, {} step would be performed".format(step))
            for i in range(step):
                self.output, self.phi, self.disp = self.forward()
                self.input = torch.cat((self.output,self.target),1)
                self.warped_label_map = self.get_warped_label_map(self.l_moving, self.phi)
                self.l_moving = self.warped_label_map

            warped_label_map_np  =self.warped_label_map.detach().cpu().numpy()
            self.l_target_np = self.l_target.detach().cpu().numpy()
            self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np, rm_bg=False)






    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)



    def save_fig_3D(self,phase):
        saving_folder_path = os.path.join(self.record_path, '3D')
        make_dir(saving_folder_path)
        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_"+phase+ "_iter_" + str(self.iter_count)
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

    def save_fig_2D(self,phase):
        saving_folder_path = os.path.join(self.record_path, '2D')
        make_dir(saving_folder_path)

        for i in range(self.moving.size(0)):
            appendix = self.fname_list[i] + "_"+phase+"_iter_" + str(self.iter_count)
            save_image_with_scale(saving_folder_path + '/' + appendix + "_moving.tif", self.moving[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_target.tif", self.target[i, 0, ...])
            save_image_with_scale(saving_folder_path + '/' + appendix + "_reproduce.tif", self.output[i, 0, ...])

    def save_fig(self,phase,standard_record=False,saving_gt=True):
        from model_pool.visualize_registration_results import  show_current_images
        visual_param={}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = self.record_path
        visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 8
        visual_param['pair_path'] = self.fname_list
        visual_param['iter'] = phase+"_iter_" + str(self.iter_count)
        disp=None
        extra_title = 'disp'
        if self.disp is not None and len(self.disp.shape)>2 and not self.demon_on:
            disp = ((self.disp[:,...]**2).sum(1))**0.5


        if self.demon_on and self.disp is not None:
            disp = self.disp[:,0,...]
            extra_title='affine'
        show_current_images(self.iter_count,  self.moving, self.target,self.output, self.l_moving,self.l_target,self.warped_label_map,
                            disp, extra_title, self.phi, visual_param=visual_param)


    def set_val(self):
        self.is_train = False

    def set_debug(self):
        self.is_train = False

    def set_test(self):
        self.is_train = False



