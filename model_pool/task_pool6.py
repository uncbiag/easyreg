import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))

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







#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_261_vonet_ce_custom')
# dm = DataTask('task_261_vonet_ce_custom')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='brats'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['old_gpu_ids']=3
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = ''#"/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/task_261_vonet_ce_custom/checkpoints/epoch_240_"
# dm.data_par['datapro']['dataset']['task_name']='brats_com_full_ff_th01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.0
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 4  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# dm.data_par['datapro']['seg']['use_org_size']= False
# dm.data_par['datapro']['seg']['detect_et']= True
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'task_0.05_prior_net'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='prior_net'
# tsm.task_par['tsk_set']['epoch'] = 300
# tsm.task_par['tsk_set']['model'] = 'pnet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 200
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.1   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet')
# dm = DataTask('task_263_vonet')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='brats'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['dataset_name'] = dm.data_par['datapro']['dataset']['dataset_name']
# tsm.task_par['tsk_set']['train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 1e-3 ###########################
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 2  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/resid_auto_wlimit_logp_limit_light4_bz4/checkpoints/epoch_50_"
# dm.data_par['datapro']['dataset']['task_name']='brats_com_full_ff_th01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 4  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.05 #######################################33
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]   #######################################
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'residauto_wlimit_logp_limit_light1_bz4_cropratio0.05'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='gb_net'
# tsm.task_par['tsk_set']['epoch'] = 360
# tsm.task_par['tsk_set']['model'] = 'gbnet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10
#
#
# tsm.task_par['tsk_set']['gbnet_model_s'] =0
# #tsm.task_par['tsk_set']['gbnet_model_e'] =3
# tsm.task_par['tsk_set']['auto_context'] =True
# tsm.task_par['tsk_set']['adaboost'] =False
# tsm.task_par['tsk_set']['residual'] =not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['end2end'] =True
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = not tsm.task_par['tsk_set']['adaboost'] or tsm.task_par['tsk_set']['end2end']
# tsm.task_par['tsk_set']['update_model_by_val'] = not tsm.task_par['tsk_set']['adaboost']
# tsm.task_par['tsk_set']['tor_thre'] = 1 if tsm.task_par['tsk_set']['adaboost'] and not tsm.task_par['tsk_set']['end2end']  else 0.02
# tsm.task_par['tsk_set']['update_model_epoch_torl']= 50 if not tsm.task_par['tsk_set']['end2end'] else 2000
#
# tsm.task_par['tsk_set']['update_model_torl'] = 2
# debug_num = 4 if tsm.task_par['tsk_set']['adaboost'] else 1
#
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,debug_num]
# tsm.task_par['tsk_set']['optim']['lr'] = 1e-3
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*100
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
















#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_261_vonet_wce_bg_avg')
# dm = DataTask('task_261_vonet_wce_bg_avg')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='brats'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['mode_train'] = False  ########################################3333
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 1e-4
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = ""#""/playpen/zyshen/data/lpba_2_vonet_lpba_seg_nopatchedmy_balanced_random_crop/task_262_vonet_wce/checkpoints/epoch_160_"
# tsm.task_par['tsk_set']['vonet_test_path']= '/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/task_261_vonet_wce_bg_avg/checkpoints'
# dm.data_par['datapro']['dataset']['task_name']='brats_com_full_ff_th01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 4  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.1
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'task_261_vonet_wce_bg_avg_test_asmbg025'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_f'
# tsm.task_par['tsk_set']['epoch'] = 280
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 5
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['only_bg_avg_update'] = True
# tsm.task_par['tsk_set']['loss']['only_bg_avg_log_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 180
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(182,280,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*1000
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#




#
################ Task 0   input -1 1#############
tsm = ModelTask('task_262_vonet_0.2imd_fix')
dm = DataTask('task_262_vonet_0.2imd_fix')
dm.data_par['datapro']['task_type']='seg'
dm.data_par['datapro']['dataset']['dataset_name']='brats'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
tsm.task_par['tsk_set']['train'] = False
tsm.task_par['tsk_set']['mode_train'] = False  ########################################3333

tsm.task_par['tsk_set']['dg_key_word'] = ''
tsm.task_par['tsk_set']['save_by_standard_label'] = True
tsm.task_par['tsk_set']['continue_train'] =False
tsm.task_par['tsk_set']['old_gpu_ids']=2
tsm.task_par['tsk_set']['gpu_ids'] = 2  #1

tsm.task_par['tsk_set']['model_path'] =''#"/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/task_261_vonet_imd_fix_avg_bg/checkpoints/epoch_35_"
tsm.task_par['tsk_set']['vonet_test_path']= '/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/task_261_vonet_imd_fix_avg_bg/checkpoints'

dm.data_par['datapro']['dataset']['task_name']='brats_com_full_ff_th01'
dm.data_par['datapro']['dataset']['prepare_data']=False
dm.data_par['datapro']['seg']['sched']='nopatched'


tsm.task_par['tsk_set']['n_in_channel'] = 4  #1
dm.data_par['datapro']['seg']['add_resampled']= False
dm.data_par['datapro']['seg']['add_loc']= False
tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']

dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.1


dm.data_par['datapro']['seg']['save_train_custom']=True
dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
dm.data_par['datapro']['seg']['partition']['flicker_on']=False
dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
dm.data_par['datapro']['seg']['partition']['flicker_range']=5

tsm.task_par['tsk_set']['task_name'] = 'task_261_vonet_imd_fix_avg_bg_asmbg025'  #task42_unet4_base
tsm.task_par['tsk_set']['network_name'] ='UNet_asm_f'
tsm.task_par['tsk_set']['epoch'] = 280
tsm.task_par['tsk_set']['model'] = 'vonet'
tsm.task_par['tsk_set']['batch_sz'] = 2
tsm.task_par['tsk_set']['val_period'] =10
tsm.task_par['tsk_set']['loss']['update_epoch'] =5
tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= True

tsm.task_par['tsk_set']['loss']['type'] = 'ce_imd'
tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
tsm.task_par['tsk_set']['loss']['log_update'] = False
tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
tsm.task_par['tsk_set']['loss']['continuous_update'] = False
tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0


tsm.task_par['tsk_set']['voting']['start_saving_model'] = 180
tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(182,280,4))


tsm.task_par['tsk_set']['criticUpdates'] = 2
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*1000
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases


tsm.save()
dm.save()
run_one_task()
#





# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_262_vonet_wce')
# dm = DataTask('task_262_vonet_wce')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='brats'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['mode_train'] = False  ########################################3333
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['old_gpu_ids']=1
# tsm.task_par['tsk_set']['gpu_ids'] = 1  #1
#
# tsm.task_par['tsk_set']['model_path'] =''# "/playpen/zyshen/data/lpba_2_vonet_lpba_seg_nopatchedmy_balanced_random_crop/task_262_vonet_t9_wce/checkpoints/epoch_200_"
# tsm.task_par['tsk_set']['vonet_test_path']= '/playpen/zyshen/data/brats_com_full_ff_th01_brats_seg_nopatchedmy_balanced_random_crop/task_261_vonet_t9_wce_avg_bg/checkpoints'
#
# dm.data_par['datapro']['dataset']['task_name']='brats_com_full_ff_th01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 4  #1
# dm.data_par['datapro']['seg']['add_resampled']= False
# dm.data_par['datapro']['seg']['add_loc']= False
# tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']
#
# dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
# dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.1
#
#
# dm.data_par['datapro']['seg']['save_train_custom']=True
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=False
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'task_261_vonet_t9_wce_avg_bg_test'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_t9_con'
# tsm.task_par['tsk_set']['epoch'] = 280
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =10
# tsm.task_par['tsk_set']['loss']['update_epoch'] =5
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['only_bg_avg_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 180
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(182,280,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*10000
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()

