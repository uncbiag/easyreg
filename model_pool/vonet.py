import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from glob import glob
from .vonet_pool import UNet_asm
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

class Vonet(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        n_in_channel = 4
        self.n_class = opt['tsk_set']['extra_info']['num_label']
        self.standard_label = opt['tsk_set']['extra_info']['standard_label_index']
        self.save_by_standard_label =opt['tsk_set']['save_by_standard_label']
        which_epoch = opt['tsk_set']['which_epoch']
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        self.loss_update_epoch = opt['tsk_set']['loss']['update_epoch']
        self.activate_epoch = opt['tsk_set']['loss']['activate_epoch']
        self.imd_weighted_loss_on =  opt['tsk_set']['loss']['imd_weighted_loss_on']
        self.start_saving_model =  opt['tsk_set']['voting']['start_saving_model']
        self.saving_voting_per_epoch =  opt['tsk_set']['voting']['saving_voting_per_epoch']
        self.voting_save_sched =opt['tsk_set']['voting']['voting_save_sched']
        tile_sz =  opt['dataset']['tile_size']
        overlap_size = opt['dataset']['overlap_size']
        padding_mode = opt['dataset']['padding_mode']
        self.network = UNet_asm(n_in_channel, self.n_class)
        #self.network = CascadedModel([UNet_light1(n_in_channel,self.n_class,bias=True,BN=True)]+[UNet_light1(n_in_channel+self.n_class,self.n_class,bias=True,BN=True) for _ in range(3)],end2end=True, auto_context=True,residual=True)
        #self.network.apply(unet_weights_init)
        # self.network = VNet(n_in_channel, self.n_class)
        # self.network.apply(vnet_weights_init)
        # if self.continue_train:
        #     self.load_network(self.network,'unet',which_epoch)
        self.loss_fn = Loss(opt)
        self.loss_update_count = 0.
        self.loss_buffer = np.zeros([1, self.n_class])
        self.optimizer_fea, self.lr_scheduler_fea, self.exp_lr_scheduler_fea =self.init_optim(opt['tsk_set']['optim'])
        self.optimizer_dis, self.lr_scheduler_dis, self.exp_lr_scheduler_dis =self.init_optim(opt['tsk_set']['optim'])
        self.partition = Partition(tile_sz, overlap_size, padding_mode)
        self.res_record = {'gt':{}}
        check_point_path = opt['tsk_set']['path']['check_point_path']
        self.asm_path = os.path.join(check_point_path,'asm_models')
        make_dir(self.asm_path)
        self.start_asm_learning = False
        self.check_point_path = opt['tsk_set']['path']['check_point_path']
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


    def cal_loss(self,output= None):
        """"
        output should be B x n_class x ...
        gt    should be B x 1 x.......
        """
        if self.imd_weighted_loss_on:
            self.get_imd_weight_loss()
        output = self.output if output is None else output
        return self.loss_fn.get_loss(output,self.gt)

    def cal_seq_loss(self,output_seq):
        loss =0.0
        for output in output_seq:
            loss += self.cal_loss(output)
        return loss

    def set_cur_epoch(self,epoch):
        self.cur_epoch = epoch
        self.cur_epoch_beg_tag = True




    # get image paths
    def get_image_paths(self):
        return self.fname_list



    def backward_net(self):
        self.loss.backward()




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
            self.start_asm_learning = True





    def auto_saving_model(self):
        if self.voting_save_sched == 'default':
            log = self.cur_epoch > self.start_saving_model and self.cur_epoch % self.saving_voting_per_epoch==0
            log = log and self.cur_epoch_beg_tag
            self.cur_epoch_beg_tag = False
            return log


    def get_current_errors(self):
        return self.loss.data[0]

    def get_assamble_pred(self,split_size=4, old_verison=False):
        output = []
        if old_verison:
            self.input = torch.unsqueeze(torch.squeeze(self.input),1)
        else:
            self.input = torch.squeeze(self.input,0)
        input_split = torch.split(self.input, split_size=split_size)
        for input in input_split:
            res = self.forward(input)
            if isinstance(res,list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched =  torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data,1)[1]
        self.input= None
        self.output = self.partition.assemble(pred_patched, self.img_sz)
        self.gt_np= self.gt.data.cpu().numpy()
        for i in range(self.gt_np.shape[0]):
            fname = self.fname_list[i]
            if fname in self.res_record:
                self.res_record[fname].append((self.output,self.iter_count))
            else:
                self.res_record[fname] =[(self.output,self.iter_count)]
                self.res_record['gt'][fname] =[self.gt_np[0]]





    def get_evaluation(self,brats_eval_on=True):
        print("start evaluate img{}".format(self.fname_list))
        self.gt = None
        self.val_res_dic = get_multi_metric(np.expand_dims(self.output,0), self.gt_np,rm_bg=False) # the correct version maybe np.expand_dims(np.expand_dims(self.output,0))
        if not self.print_val_detail:
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))
        else:
            print('batch_avg_res{}'.format(self.val_res_dic['batch_avg_res']))
            print('batch_label_avg_res:{}'.format(self.val_res_dic['batch_label_avg_res']))
        if brats_eval_on:
            output = self.output.copy()
            gt = self.gt_np.copy()
            output[np.where(output!=0)]=1
            gt[np.where(gt!=0)]=1
            val_res_dic = get_multi_metric(np.expand_dims(output,0),gt,rm_bg=True)
            print('here is the WT :{}'.format(val_res_dic['batch_label_avg_res']))
            output = self.output.copy()
            gt = self.gt_np.copy()
            output[np.where((output==3)|(output ==1))] = 1
            output[np.where(output == 2)] = 0
            gt[np.where((gt == 3) | (gt == 1))] = 1
            gt[np.where(gt == 2)] = 0
            val_res_dic = get_multi_metric(np.expand_dims(output, 0), gt, rm_bg=True)
            print('here is the TC :{}'.format(val_res_dic['batch_label_avg_res']))
            output = self.output.copy()
            gt = self.gt_np.copy()
            output[np.where(output !=3)] = 0
            output[np.where((output == 3))] = 1
            gt[np.where(gt!= 3)] = 0
            gt[np.where((gt==3))] = 1
            val_res_dic = get_multi_metric(np.expand_dims(output, 0), gt, rm_bg=True)
            print('here is the ET :{}'.format(val_res_dic['batch_label_avg_res']))

    def get_output_map(self,split_size=4):
        output = []
        self.input = torch.squeeze(self.input,0)
        input_split = torch.split(self.input, split_size=split_size)
        for input in input_split:
            output.append(self.forward(input).cpu())
        pred_patched =  torch.cat(output, dim=0)
        print(pred_patched.shape)
        self.input= None
        self.output = self.partition.assemble(pred_patched.data[:,0], self.img_sz)
        self.gt_np = self.gt.data.cpu().numpy()


    def cal_val_errors(self, split_size=2):
        self.cal_test_errors(split_size)

    def cal_test_errors(self,split_size=2):
        self.get_assamble_pred(split_size)
        self.get_evaluation()
    def get_pred_img(self,split_size=2):
        self.get_assamble_pred(split_size)




    def update_loss(self, epoch, end_of_epoch):
        if self.loss_update_epoch>0 and epoch>=self.activate_epoch and epoch % self.loss_update_epoch ==0:
            dice_h_shape = self.val_res_dic['batch_avg_res']['dice'].shape
            loss_buffer_shape = self.loss_buffer.shape
            if dice_h_shape == loss_buffer_shape:
                self.loss_update_count += 1.0
                self.loss_buffer +=self.val_res_dic['batch_avg_res']['dice']
            else:
                print("Warning the valdiation size is not the same {}, compared with{}, skip....".format(dice_h_shape, loss_buffer_shape))
            if end_of_epoch:
                resid = 1- self.loss_buffer/self.loss_update_count
                resid = resid/np.sum(resid)   #from tsk31 #fromtsk46_4 # tsk51
                self.opt['tsk_set']['loss']['residue_weight'] = resid
                self.opt['tsk_set']['loss']['residue_weight_gama'] = sigmoid_explode(epoch % 10, static=1, k=4)
                self.opt['tsk_set']['loss']['residue_weight_alpha'] = sigmoid_decay(epoch % 10, static=1, k=4)
                record_weight = self.loss_fn.record_weight
                self.loss_fn = Loss(self.opt,record_weight)
                self.loss_update_count =0.
                self.loss_buffer = np.zeros([1, self.n_class])




    def get_imd_weight_loss(self):
        if self.cur_epoch >= self.activate_epoch:
            output = torch.max(self.output.data,1)[1]
            output = output.cpu().numpy()
            metric_res= get_multi_metric(output, self.gt.cpu().data.numpy())
            dice_weights = metric_res['batch_avg_res']['dice']
            label_list = metric_res['label_list']
            log_resid_dice_weights = np.log1p(1. - dice_weights)
            log_resid_dice_weights = log_resid_dice_weights / np.sum(log_resid_dice_weights)  #
            weights = np.zeros(self.n_class)
            weights[label_list] = log_resid_dice_weights
            weights = torch.cuda.FloatTensor(weights)
            self.loss_fn = Loss(self.opt, record_weight=None, imd_weight=weights)


    def get_val_res(self):
        return self.val_res_dic['batch_label_avg_res']['dice']

    def get_test_res(self):
        return self.get_val_res()


    def get_current_visuals(self):
        return OrderedDict([('input', self.input), ('output', self.output)])

    def save(self, label):
        self.save_network(self.network, 'unet', label, self.gpu_ids)

    def get_standard_label(self):
        for ind in range(self.n_class):
            self.gt_np[np.where(self.gt_np==ind)]=self.standard_label[ind]
            self.output[np.where(self.output==ind)]=self.standard_label[ind]


    def resize_into_orign_size(self,input):
        output = np.zeros(self.origin_size)
        if input.shape < self.origin_size:
            pading_size = np.array(self.origin_size) - np.array(input.shape)
            before_id = (pading_size + 1)//2
            after_id = before_id + np.array(input.shape)
            output[before_id[0]:after_id[0],before_id[1]:after_id[1],before_id[2]:after_id[2]] = input.copy()
        else:
            pading_size = np.array(input.shape)-np.array(self.origin_size)
            before_id = (pading_size + 1) // 2
            after_id = before_id + np.array(input.shape)
            output = input[before_id[0]:after_id[0], before_id[1]:after_id[1], before_id[2]:after_id[2]].copy()
        return output





    def save_fig(self,phase,standard_record=True,saving_gt=True):
        saving_folder_path = os.path.join(self.record_path, 'output')
        make_dir(saving_folder_path)
        if self.save_by_standard_label:
            self.get_standard_label()
        for i in range(self.gt_np.shape[0]):
            itk_image = self.get_file_info( self.fname_list[i])
            appendix = self.fname_list[i] + "_"+phase+ "_iter_" + str(self.iter_count)
            if standard_record==False:
                saving_file_path = saving_folder_path + '/' + appendix + "_output.nii.gz"
            else:
                file_name = self.fname_list[i].split('_f')[0]
                saving_file_path = saving_folder_path + '/' + file_name + ".nii.gz"
            output = self.resize_into_orign_size(self.output)
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(itk_image.GetSpacing())
            output.SetOrigin(itk_image.GetOrigin())
            output.SetDirection(itk_image.GetDirection())
            sitk.WriteImage(output, saving_file_path)
            if saving_gt:
                appendix = self.fname_list[i] + "_" + phase + "_iter_" + str(self.iter_count)
                saving_file_path = saving_folder_path + '/' + appendix + "_gt.nii.gz"
                output = self.resize_into_orign_size(np.squeeze(self.gt_np).astype(np.int32))
                output = sitk.GetImageFromArray(output)
                output.SetSpacing(itk_image.GetSpacing())
                output.SetOrigin(itk_image.GetOrigin())
                output.SetDirection(itk_image.GetDirection())
                sitk.WriteImage(output, saving_file_path)

    def get_file_info(self,file_name):
        # sample_label_path = self.opt['tsk_set']['extra_info']['sample_label_path']
        # file_name =file_name.split('_t')[0]
        # label_folder_ch= file_name.split('_f')[0]
        # label_folder_par_path, label_post = os.path.split(os.path.split(sample_label_path)[0])[0],os.path.split(sample_label_path)[1].split('.',1)[1]
        # file_label_path = os.path.join(os.path.join(label_folder_par_path,label_folder_ch),file_name+'.'+label_post)
        # image = sitk.ReadImage(file_label_path)
        sample_label_path = self.opt['tsk_set']['extra_info']['sample_data_path']
        raw_data_path = self.opt['dataset']['raw_data_path']
        file_name = file_name.split('_t')[0]
        from os.path import  join
        label_post = os.path.split(sample_label_path)[1].split('.',1)[1]
        f_path = join(raw_data_path, '**', file_name+'.'+label_post)
        f_filter = glob(f_path, recursive=True)
        if len(f_filter)==0:
            f_path=f_path.replace('/playpen','/playpen/raid')
            f_filter = glob(f_path, recursive=True)
        if len(f_filter)==1:
            image = sitk.ReadImage(f_filter[0])
        else:
            print("Warning, the source file is not founded during file saving, default info from {} is used".format(sample_label_path))
            sample_label_path = self.opt['tsk_set']['extra_info']['sample_label_path']
            image = sitk.ReadImage(sample_label_path)
        self.origin_size = sitk.GetArrayFromImage(image).shape
        return image

    def get_period_voting_map(self,period_record_dic, show_period_result=True):
        """
        :param period_record_dic:  dict, include the filename:(ith_output_map, ith_iter)
        :return: dic, include the filename: voting_map
        """
        import pickle
        saving_folder_path = os.path.join(self.record_path, 'voting')
        make_dir(saving_folder_path)
        with open(saving_folder_path+'/' + 'period_res_record' + '.pkl', 'wb') as f:
            pickle.dump(period_record_dic, f, pickle.HIGHEST_PROTOCOL)
        for key, value in period_record_dic.items():
            multi_period_res = [period_res[0] for period_res in value]
            multi_period = [period_res[1] for period_res in value]
            multi_period_res = np.stack(multi_period_res, 0)
            period_voting_map = np.max(multi_period_res, 0)
            gt = period_record_dic['gt'][key]
            if show_period_result:
                for i, period in enumerate(multi_period):
                    print("the current period is {}".format(period))
                    val_res_dic = get_multi_metric(np.expand_dims(multi_period_res[i], 0), gt,
                                               rm_bg=False)
                    print("the result of file{} during period{} is:".format(key, period))
                    print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']))
                    print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']))
            val_res_dic = get_multi_metric(np.expand_dims(period_voting_map, 0), gt,rm_bg=False)  # the correct version maybe np.expand_dims(np.expand_dims(self.output,0))
            print('the voting result of file {} during period {}:'.format(key,multi_period))
            print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']))
            print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']))
            print()
            itk_image = self.get_file_info(key)
            appendix = key+ "_voting"
            saving_file_path = saving_folder_path + '/' + appendix + "_output.nii.gz"
            output = self.resize_into_orign_size(period_voting_map)
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(itk_image.GetSpacing())
            output.SetOrigin(itk_image.GetOrigin())
            output.SetDirection(itk_image.GetDirection())
            sitk.WriteImage(output, saving_file_path)











