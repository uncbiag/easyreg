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
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet')
# dm = DataTask('task_263_vonet')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['continue_train_lr'] = 2e-4 ###########################
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 3  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet/checkpoints/epoch_200_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_vonet'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_f'
# tsm.task_par['tsk_set']['epoch'] = 360
# tsm.task_par['tsk_set']['model'] = 'vonet'
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
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,4]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#




#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet_wce')
# dm = DataTask('task_263_vonet_wce')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['continue_train_lr'] = 4e-4
#
# tsm.task_par['tsk_set']['old_gpu_ids']=1
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = ""#""/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet_wce/checkpoints/epoch_200_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_vonet_wce'  #0.15
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_f'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 4
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 120
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(150,200,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,4]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#





#
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet_t9')
# dm = DataTask('task_263_vonet_t9')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 1  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet_t9/checkpoints/epoch_210_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_vonet_t9'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_t9_con'
# tsm.task_par['tsk_set']['epoch'] = 360
# tsm.task_par['tsk_set']['model'] = 'vonet'
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
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,4]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
# print("hello, here is t9")
#
#

#
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet_t9_wce')
# dm = DataTask('task_263_vonet_t9_wce')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['mode_train'] = False
#
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['continue_train_lr'] = 2e-5
#
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = ''#"/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet_t9_wce/checkpoints/epoch_185_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_vonet_t9_wce'  #0.15
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_t9_con'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 4
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 120
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(122,200,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet_t9_wce')
# dm = DataTask('task_263_vonet_t9_wce')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['mode_train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['continue_train_lr'] = 2e-4
#
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 2  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet_t9_wce/checkpoints/epoch_200_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_vonet_t9_wce'  #0.15
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_t9_con'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 4
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 120
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# #tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(122,200,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
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
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_2_vonet')
# dm = DataTask('task_263_2_vonet')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 2e-4 ###########################
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet/checkpoints/epoch_200_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_2_vonet_sprelu'  #task42_unet4_base
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_sim_prelu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['model'] = 'vonet'
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
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 200
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#




#
#
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_2_vonet_wce')
# dm = DataTask('task_263_2_vonet_wce')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 4e-4
#
# tsm.task_par['tsk_set']['old_gpu_ids']=1
# tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
#
# tsm.task_par['tsk_set']['model_path'] = ""#""/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_263_vonet_wce/checkpoints/epoch_200_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'task_263_2_vonet_sprelu_wce'  #0.15
# tsm.task_par['tsk_set']['network_name'] ='UNet_asm_sim_prelu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['model'] = 'vonet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 4
# tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 120
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
# tsm.task_par['tsk_set']['voting']['epoch_list'] = list(range(150,200,4))
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,0]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*5
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
# ################ Task 0   input -1 1#############
# tsm = ModelTask('task_263_vonet')
# dm = DataTask('task_263_vonet')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = True
# tsm.task_par['tsk_set']['dg_key_word'] = ''
# tsm.task_par['tsk_set']['save_by_standard_label'] = True
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['continue_train_lr'] = 5e-4 ###########################
# tsm.task_par['tsk_set']['old_gpu_ids']=0
# tsm.task_par['tsk_set']['gpu_ids'] = 1  #1
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/task_ada_4_0.1_acfalse_light3_th.5/checkpoints/epoch_125_"
# dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['sched']='nopatched'
#
#
# tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
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
# tsm.task_par['tsk_set']['task_name'] = 'debug_4'  #task42_unet4_base
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
# tsm.task_par['tsk_set']['loss']['ce']['reduced'] = False
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
# dm.data_par['datapro']['seg']['use_org_size']= False
#
# tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
# tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2
#
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,4,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*10
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases
#
#
# tsm.save()
# dm.save()
# run_one_task()
#





################ Task 0   input -1 1#############
tsm = ModelTask('task_263_vonet')
dm = DataTask('task_263_vonet')
dm.data_par['datapro']['task_type']='seg'
dm.data_par['datapro']['dataset']['dataset_name']='oai'
dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
tsm.task_par['tsk_set']['train'] = True
tsm.task_par['tsk_set']['dg_key_word'] = ''
tsm.task_par['tsk_set']['save_by_standard_label'] = True
tsm.task_par['tsk_set']['continue_train'] =True
tsm.task_par['tsk_set']['continue_train_lr'] = 5e-4 ###########################
tsm.task_par['tsk_set']['old_gpu_ids']=0
tsm.task_par['tsk_set']['gpu_ids'] = 0  #1

tsm.task_par['tsk_set']['model_path'] = "/playpen/raid/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop/debug_3/checkpoints/epoch_40_"
dm.data_par['datapro']['dataset']['task_name']='oai_2_vnet'
dm.data_par['datapro']['dataset']['prepare_data']=False
dm.data_par['datapro']['seg']['sched']='nopatched'


tsm.task_par['tsk_set']['n_in_channel'] = 1  #1
dm.data_par['datapro']['seg']['add_resampled']= False
dm.data_par['datapro']['seg']['add_loc']= False
tsm.task_par['tsk_set']['add_resampled']= dm.data_par['datapro']['seg']['add_resampled']

dm.data_par['datapro']['seg']['num_crop_per_class_per_train_img']=-1
dm.data_par['datapro']['seg']['transform']['transform_seq']=['my_balanced_random_crop']
dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio']= 0.1


dm.data_par['datapro']['seg']['save_train_custom']=True
dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
dm.data_par['datapro']['seg']['partition']['overlap_size']=[0,0,0]
dm.data_par['datapro']['seg']['partition']['flicker_on']=False
dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
dm.data_par['datapro']['seg']['partition']['flicker_range']=5

tsm.task_par['tsk_set']['task_name'] = 'debug_3'  #task42_unet4_base
tsm.task_par['tsk_set']['network_name'] ='gb_net'
tsm.task_par['tsk_set']['epoch'] = 360
tsm.task_par['tsk_set']['model'] = 'gbnet'
tsm.task_par['tsk_set']['batch_sz'] = 1
tsm.task_par['tsk_set']['val_period'] =4
tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
tsm.task_par['tsk_set']['loss']['imd_weighted_loss_on']= False

tsm.task_par['tsk_set']['loss']['type'] = 'ce'
tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
tsm.task_par['tsk_set']['loss']['ce']['reduced'] = False
tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
tsm.task_par['tsk_set']['loss']['log_update'] = False
tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
tsm.task_par['tsk_set']['loss']['continuous_update'] = False
tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.3
tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
tsm.task_par['tsk_set']['loss']['activate_epoch'] = 10


tsm.task_par['tsk_set']['gbnet_model_s'] =1
dm.data_par['datapro']['seg']['use_org_size']= True

tsm.task_par['tsk_set']['voting']['start_saving_model'] = 300
tsm.task_par['tsk_set']['voting']['saving_voting_per_epoch'] = 2


tsm.task_par['tsk_set']['criticUpdates'] = 1
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,4,1]
tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*10
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5   # the learning rate should be ajusted in brats cases


tsm.save()
dm.save()
run_one_task()
