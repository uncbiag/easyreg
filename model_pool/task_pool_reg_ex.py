import matplotlib as matplt
matplt.use('Agg')
import torch

import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
#os.environ["CUDA_CACHE_PATH"] = "/playpen/zhenlinx/.cuda_cache
from model_pool.global_variable import *
import data_pre.module_parameters as pars
from abc import ABCMeta, abstractmethod
from model_pool.piplines import run_one_task
class BaseTask():
    __metaclass__ = ABCMeta
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def save(self):
        pass

class DataTask(BaseTask):
    def __init__(self,name):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON('../settings/base_data_settings.json')


    def save(self):
        self.data_par.write_ext_JSON('../settings/data_settings.json')

class ModelTask(BaseTask):
    def __init__(self,name):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON('../settings/base_task_settings.json')

    def save(self):
        self.task_par.write_ext_JSON('../settings/task_settings.json')



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  NiftyReg-affine

#
# ################ Task 0   input -1 1#############
# for sess in ['intra','inter']:
#
#     tsm = ModelTask('task_reg')
#     dm = DataTask('task_reg')
#     is_llm = False
#     dm.data_par['datapro']['task_type']='reg'
#     dm.data_par['datapro']['dataset']['dataset_name']='oai'
#     dm.data_par['datapro']['reg']['sched']= sess
#     dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
#     dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
#     # tsm.task_par['tsk_set']['save_fig_on'] = False
#     tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
#     tsm.task_par['tsk_set']['low_res_factor'] =0.5
#     tsm.task_par['tsk_set']['train'] = False
#     dm.data_par['datapro']['reg']['test_fail_case'] = False
#
#     tsm.task_par['tsk_set']['dg_key_word'] = ''
#     tsm.task_par['tsk_set']['save_by_standard_label'] = True
#     tsm.task_par['tsk_set']['continue_train'] =False
#     tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
#     tsm.task_par['tsk_set']['old_gpu_ids']=2
#     tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
#     tsm.task_par['tsk_set']['model_path'] =''#'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
#     #                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',
#
#     if is_llm:
#         dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#         tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
#     dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
#     dm.data_par['datapro']['dataset']['prepare_data']=False
#     dm.data_par['datapro']['seg']['sched']='nopatched'
#     dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
#     tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
#     dm.data_par['datapro']['seg']['add_resampled']= False
#     dm.data_par['datapro']['seg']['add_loc']= False
#     tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
#     dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
#     dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
#     dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
#     dm.data_par['datapro']['seg']['save_train_custom']=True
#     dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
#     dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
#     dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#     dm.data_par['datapro']['seg']['partition']['flicker_on']=False
#     dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
#     dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
#     tsm.task_par['tsk_set']['task_name'] = 'run_niftyreg_affine_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
#     tsm.task_par['tsk_set']['network_name'] ='affine'  #'mermaid' 'svf' 'syn' affine bspline
#     tsm.task_par['tsk_set']['epoch'] = 300 #300
#     tsm.task_par['tsk_set']['model'] = 'nifty_reg'  #mermaid_iter reg_net  ants  nifty_reg
#     tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
#     tsm.task_par['tsk_set']['val_period'] =10
#     tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#     tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
#     tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
#     tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['log_update'] = False
#     tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
#     tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['continuous_update'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
#     tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
#     tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
#     tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
#     #tsm.task_par['tsk_set']['gbnet_model_e'] =1
#     tsm.task_par['tsk_set']['auto_context'] =False
#     tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
#     tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
#     tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
#     tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
#     tsm.task_par['tsk_set']['update_model_torl'] = 2
#     tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
#     debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
#     dm.data_par['datapro']['seg']['use_org_size']= False
#
#     tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
#     tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
#     tsm.task_par['tsk_set']['criticUpdates'] = 1
#     tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
#     tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
#     tsm.save()
#     dm.save()
#     run_one_task()
#
#
#
#
#















#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  affine-opt
#
#
# ################ Task 0   input -1 1#############
# extra_setting = [4,2]
# for i,sess in enumerate(['intra','inter']):
#     tsm = ModelTask('task_reg')
#     dm = DataTask('task_reg')
#     is_llm = False
#     dm.data_par['datapro']['task_type']='reg'
#     dm.data_par['datapro']['dataset']['dataset_name']='oai'
#     dm.data_par['datapro']['reg']['sched']= sess
#     dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
#     dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
#     # tsm.task_par['tsk_set']['save_fig_on'] = False
#     tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
#     tsm.task_par['tsk_set']['low_res_factor'] =0.5
#     tsm.task_par['tsk_set']['train'] = False
#     dm.data_par['datapro']['reg']['test_fail_case'] = False
#
#     tsm.task_par['tsk_set']['dg_key_word'] = ''
#     tsm.task_par['tsk_set']['save_by_standard_label'] = True
#     tsm.task_par['tsk_set']['continue_train'] =False
#     tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
#     tsm.task_par['tsk_set']['old_gpu_ids']=2
#     tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
#     tsm.task_par['tsk_set']['model_path'] =''#'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
#     #                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',
#
#     # if is_llm:
#     # dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     # tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
#     dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
#     dm.data_par['datapro']['dataset']['prepare_data']=False
#     dm.data_par['datapro']['seg']['sched']='nopatched'
#     dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
#     tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
#     dm.data_par['datapro']['seg']['add_resampled']= False
#     dm.data_par['datapro']['seg']['add_loc']= False
#     tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
#     dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
#     dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
#     dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
#     dm.data_par['datapro']['seg']['save_train_custom']=True
#     dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
#     dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
#     dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#     dm.data_par['datapro']['seg']['partition']['flicker_on']=False
#     dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
#     dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
#     tsm.task_par['tsk_set']['task_name'] = 'run_baseline_svf_jacobi_new2_moreiter' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
#     tsm.task_par['tsk_set']['network_name'] ='svf'  #'mermaid' 'svf' 'syn' affine bspline
#     tsm.task_par['tsk_set']['epoch'] = 300 #300
#     tsm.task_par['tsk_set']['model'] = 'mermaid_iter'  #mermaid_iter reg_net  ants  nifty_reg
#     tsm.task_par['tsk_set']['batch_sz'] = extra_setting[i]  ######################TODO#####################
#     tsm.task_par['tsk_set']['val_period'] =10
#     tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#     tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
#     tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
#     tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['log_update'] = False
#     tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
#     tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['continuous_update'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
#     tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
#     tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
#     tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
#     #tsm.task_par['tsk_set']['gbnet_model_e'] =1
#     tsm.task_par['tsk_set']['auto_context'] =False
#     tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
#     tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
#     tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
#     tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
#     tsm.task_par['tsk_set']['update_model_torl'] = 2
#     tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
#     debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
#     dm.data_par['datapro']['seg']['use_org_size']= False
#
#     tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
#     tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
#     tsm.task_par['tsk_set']['criticUpdates'] = 1
#     tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
#     tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
#     tsm.save()
#     dm.save()
#     run_one_task()
#








# ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  affine-net
#
#
# ################ Task 0   input -1 1#############
# for sess in ['intra','inter']:
#
#     tsm = ModelTask('task_reg')
#     dm = DataTask('task_reg')
#     is_llm = False
#     dm.data_par['datapro']['task_type']='reg'
#     dm.data_par['datapro']['dataset']['dataset_name']='oai'
#     dm.data_par['datapro']['reg']['sched']= sess
#     dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
#     dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
#     # tsm.task_par['tsk_set']['save_fig_on'] = False
#     tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
#     tsm.task_par['tsk_set']['low_res_factor'] =0.5
#     tsm.task_par['tsk_set']['train'] = False
#     dm.data_par['datapro']['reg']['test_fail_case'] = False
#
#     tsm.task_par['tsk_set']['dg_key_word'] = ''
#     tsm.task_par['tsk_set']['save_by_standard_label'] = True
#     tsm.task_par['tsk_set']['continue_train'] =False
#     tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
#     tsm.task_par['tsk_set']['old_gpu_ids']=2
#     tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
#     tsm.task_par['tsk_set']['model_path'] ='/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_affine_net_sym_lncc/checkpoints/epoch_1070_' \
#         if sess == 'intra' else '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_affine_symstep5_lncc_bi/checkpoints/epoch_60_'
#     if is_llm:
#         dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#         tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
#     dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
#     dm.data_par['datapro']['dataset']['prepare_data']=False
#     dm.data_par['datapro']['seg']['sched']='nopatched'
#     dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
#     tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
#     dm.data_par['datapro']['seg']['add_resampled']= False
#     dm.data_par['datapro']['seg']['add_loc']= False
#     tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
#     dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
#     dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
#     dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
#     dm.data_par['datapro']['seg']['save_train_custom']=True
#     dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
#     dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
#     dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#     dm.data_par['datapro']['seg']['partition']['flicker_on']=False
#     dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
#     dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
#     tsm.task_par['tsk_set']['task_name'] = 'run_affine_net_sym_lncc_step7_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
#     tsm.task_par['tsk_set']['network_name'] ='affine_cycle'  #'mermaid' 'svf' 'syn' affine bspline
#     tsm.task_par['tsk_set']['epoch'] = 300 #300
#     tsm.task_par['tsk_set']['model'] = 'reg_net'  #mermaid_iter reg_net  ants  nifty_reg
#     tsm.task_par['tsk_set']['batch_sz'] = 8  ######################TODO#####################
#     tsm.task_par['tsk_set']['val_period'] =10
#     tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#     tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
#     tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
#     tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['log_update'] = False
#     tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
#     tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['continuous_update'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
#     tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
#     tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
#     tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
#     #tsm.task_par['tsk_set']['gbnet_model_e'] =1
#     tsm.task_par['tsk_set']['auto_context'] =False
#     tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
#     tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
#     tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
#     tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
#     tsm.task_par['tsk_set']['update_model_torl'] = 2
#     tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
#     debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
#     dm.data_par['datapro']['seg']['use_org_size']= False
#
#     tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
#     tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
#     tsm.task_par['tsk_set']['criticUpdates'] = 1
#     tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
#     tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
#     tsm.save()
#     dm.save()
#     run_one_task()

#
#
#
#
# ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  SYN
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_reg')
# dm = DataTask('task_reg')
# is_llm = False
# dm.data_par['datapro']['task_type']='reg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['reg']['sched']= 'inter'
# dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
# tsm.task_par['tsk_set']['low_res_factor'] =0.5
# tsm.task_par['tsk_set']['train'] = False
# dm.data_par['datapro']['reg']['test_fail_case'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
# tsm.task_par['tsk_set']['old_gpu_ids']=2
# tsm.task_par['tsk_set']['gpu_ids'] = 1  #1
#
# tsm.task_par['tsk_set']['model_path'] =''
# if is_llm:
#     dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
# dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
# dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'run_ants_refine_jacobi2' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='syn'  #'mermaid' 'svf' 'syn' affine bspline
# tsm.task_par['tsk_set']['epoch'] = 300 #300
# tsm.task_par['tsk_set']['model'] = 'ants'  #mermaid_iter reg_net  ants  nifty_reg
# tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
# tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
# #tsm.task_par['tsk_set']['gbnet_model_e'] =1
# tsm.task_par['tsk_set']['auto_context'] =False
# tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
#
#
# ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Demons
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_reg')
# dm = DataTask('task_reg')
# is_llm = False
# dm.data_par['datapro']['task_type']='reg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['reg']['sched']= 'intra'
# dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
# tsm.task_par['tsk_set']['low_res_factor'] =0.5
# tsm.task_par['tsk_set']['train'] = False
# dm.data_par['datapro']['reg']['test_fail_case'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
# tsm.task_par['tsk_set']['old_gpu_ids']=2
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] =''
# if is_llm:
#     dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
# dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
# dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'run_demons_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='demons'  #'mermaid' 'svf' 'syn' affine bspline
# tsm.task_par['tsk_set']['epoch'] = 300 #300
# tsm.task_par['tsk_set']['model'] = 'demons'  #mermaid_iter reg_net  ants  nifty_reg
# tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
# tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
# #tsm.task_par['tsk_set']['gbnet_model_e'] =1
# tsm.task_par['tsk_set']['auto_context'] =False
# tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#





#
#
#
#
#
# ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  affine-opt
#
#
# ################ Task 0   input -1 1#############
# for sess in ['intra','inter']:
#
#     tsm = ModelTask('task_reg')
#     dm = DataTask('task_reg')
#     is_llm = False
#     dm.data_par['datapro']['task_type']='reg'
#     dm.data_par['datapro']['dataset']['dataset_name']='oai'
#     dm.data_par['datapro']['reg']['sched']= sess
#     dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
#     dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
#     # tsm.task_par['tsk_set']['save_fig_on'] = False
#     tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
#     tsm.task_par['tsk_set']['low_res_factor'] =0.5
#     tsm.task_par['tsk_set']['train'] = False
#     dm.data_par['datapro']['reg']['test_fail_case'] = False
#
#     tsm.task_par['tsk_set']['dg_key_word'] = ''
#     tsm.task_par['tsk_set']['save_by_standard_label'] = True
#     tsm.task_par['tsk_set']['continue_train'] =False
#     tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
#     tsm.task_par['tsk_set']['old_gpu_ids']=2
#     tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
#     tsm.task_par['tsk_set']['model_path'] =''#'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
#     #                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',
#
#     if is_llm:
#         dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#         tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
#     dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
#     dm.data_par['datapro']['dataset']['prepare_data']=False
#     dm.data_par['datapro']['seg']['sched']='nopatched'
#     dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
#     tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
#     dm.data_par['datapro']['seg']['add_resampled']= False
#     dm.data_par['datapro']['seg']['add_loc']= False
#     tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
#     dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
#     dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
#     dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
#     dm.data_par['datapro']['seg']['save_train_custom']=True
#     dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
#     dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
#     dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#     dm.data_par['datapro']['seg']['partition']['flicker_on']=False
#     dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
#     dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
#     tsm.task_par['tsk_set']['task_name'] = 'run_niftyreg_bspline_nmi_10_jacobi_save_img' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
#     tsm.task_par['tsk_set']['network_name'] ='bspline'  #'mermaid' 'svf' 'syn' affine bspline
#     tsm.task_par['tsk_set']['epoch'] = 300 #300
#     tsm.task_par['tsk_set']['model'] = 'nifty_reg'  #mermaid_iter reg_net  ants  nifty_reg
#     tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
#     tsm.task_par['tsk_set']['val_period'] =10
#     tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#     tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
#     tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
#     tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['log_update'] = False
#     tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
#     tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['continuous_update'] = False
#     tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
#     tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#     tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
#     tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
#     tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
#     #tsm.task_par['tsk_set']['gbnet_model_e'] =1
#     tsm.task_par['tsk_set']['auto_context'] =False
#     tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
#     tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
#     tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
#     tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
#     tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
#     tsm.task_par['tsk_set']['update_model_torl'] = 2
#     tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
#     debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
#     dm.data_par['datapro']['seg']['use_org_size']= False
#
#     tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
#     tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
#     tsm.task_par['tsk_set']['criticUpdates'] = 1
#     tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
#     tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
#     tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
#     tsm.save()
#     dm.save()
#     run_one_task()
#
#






##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  affine-opt


################ Task 0   input -1 1#############
#
# tsm = ModelTask('task_reg')
# dm = DataTask('task_reg')
# is_llm = False
# dm.data_par['datapro']['task_type']='reg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['reg']['sched']= 'inter'
# dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
# tsm.task_par['tsk_set']['low_res_factor'] =0.5
# tsm.task_par['tsk_set']['train'] = False
# dm.data_par['datapro']['reg']['test_fail_case'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
# tsm.task_par['tsk_set']['old_gpu_ids']=2
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] =''#'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
# #                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',
#
# if is_llm:
#     dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
# dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
# dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'run_ants_synRA_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='syn'  #'mermaid' 'svf' 'syn' affine bspline
# tsm.task_par['tsk_set']['epoch'] = 300 #300
# tsm.task_par['tsk_set']['model'] = 'ants'  #mermaid_iter reg_net  ants  nifty_reg
# tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
# tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
# #tsm.task_par['tsk_set']['gbnet_model_e'] =1
# tsm.task_par['tsk_set']['auto_context'] =False
# tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#






##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AVSM Train


################ Task 0   input -1 1#############
tsm = ModelTask('task_reg')
dm = DataTask('task_reg')
is_llm = False
dm.data_par['datapro']['task_type']='reg'
dm.data_par['datapro']['dataset']['dataset_name']='oai'
dm.data_par['datapro']['reg']['sched']= 'inter'
dm.data_par['datapro']['reg']['is_llm'] = is_llm

dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
tsm.task_par['tsk_set']['save_fig_on'] = True
tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
tsm.task_par['tsk_set']['low_res_factor'] =0.5
tsm.task_par['tsk_set']['train'] = True
dm.data_par['datapro']['reg']['test_fail_case'] = False

tsm.task_par['tsk_set']['dg_key_word'] = ''
tsm.task_par['tsk_set']['save_by_standard_label'] = True
tsm.task_par['tsk_set']['continue_train'] =False
tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
tsm.task_par['tsk_set']['old_gpu_ids']=2
tsm.task_par['tsk_set']['gpu_ids'] = 0  #1

tsm.task_par['tsk_set']['model_path'] ='' #''/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_160_'
#                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',

if is_llm:
    dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
    tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')


dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_3000_pair' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
dm.data_par['datapro']['dataset']['prepare_data']=False
dm.data_par['datapro']['seg']['sched']='nopatched'
dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']


tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
dm.data_par['datapro']['seg']['add_resampled']= False
dm.data_par['datapro']['seg']['add_loc']= False
tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']

dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01


dm.data_par['datapro']['seg']['save_train_custom']=True
dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
dm.data_par['datapro']['seg']['partition']['flicker_on']=False
dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
dm.data_par['datapro']['seg']['partition']['flicker_range']=5

tsm.task_par['tsk_set']['task_name'] = 'train_intra_mermaid_net_1000inst_10reg_double_loss_img_coord_3scale_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
tsm.task_par['tsk_set']['network_name'] ='mermaid'  #'mermaid' 'svf' 'syn' affine bspline
tsm.task_par['tsk_set']['epoch'] = 300 #300
tsm.task_par['tsk_set']['model'] = 'reg_net'  #mermaid_iter reg_net  ants  nifty_reg
tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
tsm.task_par['tsk_set']['val_period'] =10
tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False

tsm.task_par['tsk_set']['loss']['type'] = 'ncc'
tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
tsm.task_par['tsk_set']['loss']['log_update'] = False
tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
tsm.task_par['tsk_set']['loss']['continuous_update'] = False
tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10


tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
#tsm.task_par['tsk_set']['gbnet_model_e'] =1
tsm.task_par['tsk_set']['auto_context'] =False
tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
tsm.task_par['tsk_set']['update_model_torl'] = 2
tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1

dm.data_par['datapro']['seg']['use_org_size']= False

tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2


tsm.task_par['tsk_set']['criticUpdates'] = 1
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
tsm.task_par['tsk_set']['optim']['lr'] = 1e-4            ######################TODO###################################
tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases


tsm.save()
dm.save()
run_one_task()



#
#
# # ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AVSM Test
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_reg')
# dm = DataTask('task_reg')
# is_llm = False
# dm.data_par['datapro']['task_type']='reg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['reg']['sched']= 'inter'
# dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
# tsm.task_par['tsk_set']['low_res_factor'] =0.5
# tsm.task_par['tsk_set']['train'] = False
# dm.data_par['datapro']['reg']['test_fail_case'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
# tsm.task_par['tsk_set']['old_gpu_ids']=2
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] ='/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_70_'
#     #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_90_'
#  #''/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_160_'
#
# if is_llm:
#     dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
# dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
# dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'debugging' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='mermaid'  #'mermaid' 'svf' 'syn' affine bspline
# tsm.task_par['tsk_set']['epoch'] = 300 #300
# tsm.task_par['tsk_set']['model'] = 'reg_net'  #mermaid_iter reg_net  ants  nifty_reg
# tsm.task_par['tsk_set']['batch_sz'] = 4  ######################TODO#####################
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
# tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
# #tsm.task_par['tsk_set']['gbnet_model_e'] =1
# tsm.task_par['tsk_set']['auto_context'] =False
# tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#



#
#
# ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Jacobi
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_reg')
# dm = DataTask('task_reg')
# is_llm = False
# dm.data_par['datapro']['task_type']='reg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['reg']['sched']= 'intra'
# dm.data_par['datapro']['reg']['is_llm'] = is_llm
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['input_resize_factor'] =[80./160.,192./384.,192./384]
# tsm.task_par['tsk_set']['low_res_factor'] =0.5
# tsm.task_par['tsk_set']['train'] = False
# dm.data_par['datapro']['reg']['test_fail_case'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5 ####################################################################3
# tsm.task_par['tsk_set']['old_gpu_ids']=2
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] =''
# if is_llm:
#     dm.data_par['datapro']['dataset']['output_path'] = dm.data_par['datapro']['dataset']['output_path'].replace('/playpen','/playpen/raid')
#     tsm.task_par['tsk_set']['model_path'] = tsm.task_par['tsk_set']['model_path'].replace('/playpen','/playpen/raid')
#
#
# dm.data_par['datapro']['dataset']['task_name']= 'reg_debug_labeled' #'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
# dm.data_par['datapro']['reg']['input_resize_factor'] =tsm.task_par['tsk_set']['input_resize_factor']
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.01
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'run_demons_jacobi' #'train_affine_symstep5_lncc_bi'#'run_baseline_svf_lncc_bilncc'  #'reg_mermaid_sigma2_ncc'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='demons'  #'mermaid' 'svf' 'syn' affine bspline
# tsm.task_par['tsk_set']['epoch'] = 300 #300
# tsm.task_par['tsk_set']['model'] = 'demons'  #mermaid_iter reg_net  ants  nifty_reg
# tsm.task_par['tsk_set']['batch_sz'] = 1  ######################TODO#####################
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['single_mod'] =True ############################################################3
# tsm.task_par['tsk_set']['gbnet_model_s'] =1 ############################################################3
# #tsm.task_par['tsk_set']['gbnet_model_e'] =1
# tsm.task_par['tsk_set']['auto_context'] =False
# tsm.task_par['tsk_set']['adaboost'] =True    ################################################################
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =False   ##########################################################################
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 80 if not tsm.task_par['tsk_set']['end2end'] else 2000
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3  ##############################
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
