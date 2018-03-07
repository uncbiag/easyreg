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


# #######################  Task 1  #########
# tsm = ModelTask('tsk1')
# tsm.task_par['tsk_set']['task_name'] = 'tsk1_focal_loss'
# tsm.task_par['tsk_set']['loss'] = 'focal_loss'
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.save()
# run_one_task()

# ################# Task 2  #############
# tsm = ModelTask('tsk2')
# tsm.task_par['tsk_set']['task_name'] = 'tsk2_big_batch_ce'
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 5
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/4
# tsm.save()
# run_one_task()


# ################# Task 3  #############
# tsm = ModelTask('tsk3')
#
# tsm.task_par['tsk_set']['task_name'] = 'tsk3_big_batch_focal_loss'
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['loss'] = 'focal_loss'
# tsm.task_par['tsk_set']['criticUpdates'] = 5
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/4
#
# tsm.save()
# run_one_task()

# ############## Task 4 #############
#
# tsm = ModelTask('tsk4')
# dm = DataTask('tsk4')
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='th_0.01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk4_big_batch'
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 5
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/4
#
# tsm.save()
# dm.save()
# run_one_task()


# ############## Task 5 #############
#
# tsm = ModelTask('tsk5')
# dm = DataTask('tsk5')
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=True
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#
# tsm.task_par['tsk_set']['task_name'] = 'tsk5_mid_batch'
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 6 #############
#
# tsm = ModelTask('tsk6')
# dm = DataTask('tsk6')
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='th_0.01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk6_ignore_0'
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()


# ############## Task 7 #############
#
# tsm = ModelTask('tsk7')
# dm = DataTask('tsk7')
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='th_0.01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk7_weighted_ce'
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 8 #############
#
# tsm = ModelTask('tsk8')
# dm = DataTask('tsk8')
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='th_0.01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk8_iter_200'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 9  failed  inf in focal_loss #############
#
# tsm = ModelTask('tsk9')
# dm = DataTask('tsk9')
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='th_0.01'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk9_focal_loss'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss'] = 'focal_loss'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 10 use weighted ce#############
#
# tsm = ModelTask('tsk10')
# dm = DataTask('tsk10')
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#
# tsm.task_par['tsk_set']['task_name'] = 'tsk10_iter_200_wce'
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['epoch'] = 200
#
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()


#
# ############## Task 11 #############
#
# tsm = ModelTask('tsk11')
# dm = DataTask('tsk11')
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
#
# tsm.task_par['tsk_set']['task_name'] = 'tsk11_iter_200_no_weighted_ce'
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['epoch'] = 200
#
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 12  weighted ce  input -1 1#############
# #### remeber that the normalized has been commented in read_fileio_img  and the input has been normalized to -1,1 in unet
#
# tsm = ModelTask('tsk12')
# dm = DataTask('tsk12')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=True
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task_12_iter_200_wce'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
#
# tsm.save()
# dm.save()
# run_one_task()


# ############## Task 13   input -1 1#############
# #### plateu learning rate
#
# tsm = ModelTask('tsk13')
# dm = DataTask('tsk13')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk13_plateu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'plateau'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['patience'] = 5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['threshold'] = 0.001
#
# tsm.save()
# dm.save()
# run_one_task()


# ############## Task 14 failed  input -1 1#############
#
# tsm = ModelTask('tsk14')
# dm = DataTask('tsk14')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk14_dice_loss'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'dice_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# ############## Task 15 failed   input -1 1#############
# #### plateu learning rate
#
# tsm = ModelTask('tsk15')
# dm = DataTask('tsk15')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk15_dice_loss_plateu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'dice_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'plateau'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['patience'] = 5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['threshold'] = 0.001
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 16 failed  input -1 1#############
#
# tsm = ModelTask('tsk16')
# dm = DataTask('tsk16')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk16_dice2_loss'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'dice_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# ############## Task 17  failed  input -1 1#############
# #### plateu learning rate
#
# tsm = ModelTask('tsk17')
# dm = DataTask('tsk17')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk17_dice_loss2_plateu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'dice_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'plateau'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['patience'] = 5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['threshold'] = 0.001
#
# tsm.save()
# dm.save()
# run_one_task()




# ############## Task 18   input -1 1#############
#
# tsm = ModelTask('tsk18')
# dm = DataTask('tsk18')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk18_focal_loss'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# ############## Task 19   input -1 1#############
# #### plateu learning rate
#
# tsm = ModelTask('tsk19')
# dm = DataTask('tsk19')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'tsk19_focal_loss_plateu'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'plateau'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['patience'] = 5
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['plateau']['threshold'] = 0.001
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 20#############
#  the -1 1 normalized have been removed from unet.py
# tsm = ModelTask('tsk20')
# dm = DataTask('tsk20')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# tsm.task_par['tsk_set']['task_name'] = 'tsk20_dice_loss2'   # name error
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['loss']['type'] = 'dice_loss'
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()




# ############## Task 21   input -1 1#############
#
# tsm = ModelTask('tsk21')
# dm = DataTask('tsk21')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task21_no_log'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 22   input -1 1#############
#
# tsm = ModelTask('tsk22')
# dm = DataTask('tsk22')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task22_log'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 23   input -1 1#############
#
# tsm = ModelTask('tsk23')
# dm = DataTask('tsk23')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task23_bias_bn_on'  #task23_bias_on_bn_half_on
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = -1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()



# ############## Task 24   input -1 1#############
#
# tsm = ModelTask('tsk24')
# dm = DataTask('tsk24')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task24_bias_bn_on_128'  # 'task24_bias_on_bn_half_on_128'
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = -1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()



# ############# Task 25   input -1 1#############
#
# tsm = ModelTask('tsk25')
# dm = DataTask('tsk25')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task25_bn_on_128'  # tsk25_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 10
# tsm.task_par['tsk_set']['loss']['update_epoch'] = -1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()



# ############# Task 26   input -1 1#############
#
# tsm = ModelTask('tsk26')
# dm = DataTask('tsk26')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task26_bias_on_bn__on_128'  # tsk26_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 200
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] = 10
# tsm.task_par['tsk_set']['loss']['update_epoch'] = -1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['criticUpdates'] = 3
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()




# ############# Task 27   input -1 1#############
#
# tsm = ModelTask('tsk27')
# dm = DataTask('tsk27')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task27_log_bias_on_bn_on_128'  # tsk26_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 4
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()








# ############# Task 28   input -1 1#############
#
# tsm = ModelTask('tsk28')
# dm = DataTask('tsk28')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task28_s27_static_1_val_2'  # tsk26_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()





# ############# Task 29   input -1 1#############
#
# tsm = ModelTask('tsk29')
# dm = DataTask('tsk29')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# tsm.task_par['tsk_set']['task_name'] = 'task28_s27_static_1_val_2'  # task29_log_static_1_val
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()





# ############# Task 30   input -1 1#############
#
# tsm = ModelTask('tsk30')
# dm = DataTask('tsk30')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# tsm.task_par['tsk_set']['task_name'] = 'task30_s29_only_log_residue'
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()






# ############# Task 31   input -1 1#############
#
# tsm = ModelTask('tsk31')
# dm = DataTask('tsk31')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task31_static_1_val_2'  # tsk26_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()






# ############# Task 32   input -1 1#############
#
# tsm = ModelTask('tsk32')
# dm = DataTask('tsk32')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task32_s31_momentum'  # tsk26_bias_off_bn_half_on_128
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()




# ############# Task 33   input -1 1#############
#
# tsm = ModelTask('tsk33')
# dm = DataTask('tsk33')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task33_s31_momentum_1_half_bn'
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['continuous_update'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()













#
# ############# Task 34  input -1 1############# ##  from here density fixed, gama fixed, class uniform initial, residue zero init
#
# tsm = ModelTask('tsk34')
# dm = DataTask('tsk34')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='oai'
# dm.data_par['datapro']['dataset']['task_name']='fix_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['transform']['my_bal_rand_crop']['scale_ratio'] =0.15
# dm.data_par['datapro']['seg']['patch_size']=[128,128,32]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,8]
# tsm.task_par['tsk_set']['task_name'] = 'task34_s28_density_fix'
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
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
# ############# Task 35  input -1 1#############
#
# tsm = ModelTask('tsk35')
# dm = DataTask('tsk35')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task35_s31_density_fix'
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()






# ############ Task 36  input -1 1#############
#
# tsm = ModelTask('tsk36')
# dm = DataTask('tsk36')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task36_s31_density_gama_fix'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = 2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
# ############ Task 37  input -1 1#############
#
# tsm = ModelTask('tsk37')
# dm = DataTask('tsk37')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task37_c36_base'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] = -1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()







# ############ Task 39  input -1 1#############  sigmoid epoch is 5, static 1,  during 3
#
# tsm = ModelTask('tsk39')
# dm = DataTask('tsk39')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task39_s36_add_alpha'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()









#
# ############ Task 38  input -1 1#############
#
# tsm = ModelTask('tsk38')
# dm = DataTask('tsk38')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task38_c36_base_weighted'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 5
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()





#
# ############ Task 40  input -1 1#############  sigmoid epoch is 5, static 1,  during 3
#
# tsm = ModelTask('tsk40')
# dm = DataTask('tsk40')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task40_base_res_net'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 5
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#





# ############ Task 41  input -1 1#############  sigmoid epoch is 5, static 1,  during 4
#
# tsm = ModelTask('tsk41')
# dm = DataTask('tsk41')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task41_only_resid'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] = 2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()


# # # ############## Task 42   input -1 1#############
# tsm = ModelTask('tsk42')
# dm = DataTask('tsk42')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task41_unet4_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =5
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()

#
# # # ############## Task 43   input -1 1#############
# tsm = ModelTask('tsk43')
# dm = DataTask('tsk43')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task43_unet5_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =5
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()






# # ############## Task 44   input -1 1#############
# tsm = ModelTask('tsk44')
# dm = DataTask('tsk44')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task44_unet_focal_loss_base'  # fix focal loss
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 3
# tsm.task_par['tsk_set']['val_period'] =5
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()




#
# ############## Task 45   input -1 1#############
# tsm = ModelTask('tsk45')
# dm = DataTask('tsk45')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
#
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task45_unet_focal_loss_resid'  # check focal resid is on
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['save_val_fig_epoch'] =50
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()






# # # ############## Task 46   input -1 1#############
# tsm = ModelTask('tsk46_1')
# dm = DataTask('tsk46_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task46_1_unet5B_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
# # # ############## Task 46   input -1 1#############
# tsm = ModelTask('tsk46_2')
# dm = DataTask('tsk46_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task46_2_unet5B_only_resid'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 46   input -1 1#############
# tsm = ModelTask('tsk46_3')
# dm = DataTask('tsk46_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task46_3_unet5B_focal_loss'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# # # ############## Task 46   input -1 1#############
# tsm = ModelTask('tsk46_4')
# dm = DataTask('tsk46_4')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task46_4_unet5B_focal_loss_only_resid'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
#
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [300,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
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
# # # ############## Task 47   input -1 1#############
# tsm = ModelTask('tsk47_1')
# dm = DataTask('tsk47_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task47_1_unet4B_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
# # # ############## Task 47   input -1 1#############
# tsm = ModelTask('tsk47_2')
# dm = DataTask('tsk47_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task47_2_unet4B_only_resid'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
# # # ############## Task 47   input -1 1#############
# tsm = ModelTask('tsk47_3')
# dm = DataTask('tsk47_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task47_3_unet4B_log_resid'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = True
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = False
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()

#
# # # ############## Task 47   input -1 1#############
# tsm = ModelTask('tsk47_4')
# dm = DataTask('tsk47_4')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task47_4_unet4B_focal_loss_only_resid'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
#
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()


# # # ############## Task 47   input -1 1#############
# tsm = ModelTask('tsk47_4')
# dm = DataTask('tsk47_4')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# #tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task47_4_unet4B_focal_loss_only_resid_noN'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
#
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
#
#
# tsm.save()
# dm.save()
# run_one_task()










# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_1')
# dm = DataTask('tsk48_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_1_unetB_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 1000
#
#
# tsm.save()
# dm.save()
# run_one_task()

# #
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_2')
# dm = DataTask('tsk48_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_2_unet4B_large_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_3')
# dm = DataTask('tsk48_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_3_unetB_large_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
# ## To RUN
#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_4')
# dm = DataTask('tsk48_4')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_4_unetB_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()



# # # ############## Task 49   input -1 1#############
# tsm = ModelTask('tsk49_2')
# dm = DataTask('tsk49_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task49_2_unet_t1_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
# # # ############## Task 49   input -1 1#############
# tsm = ModelTask('tsk49_3')
# dm = DataTask('tsk49_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task49_3_unet_t2_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_5')
# dm = DataTask('tsk48_5')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# #tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_5_unetB_const_lr_100'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()


# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_6')
# dm = DataTask('tsk48_6')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_6_unet3t_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task48_6_unet3t_const_lr/checkpoints/epoch_38_"
# tsm.task_par['tsk_set']['dg_key_word'] = 'input_cc_comb'
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_7')
# dm = DataTask('tsk48_7')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_7_unetbs_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_8')
# dm = DataTask('tsk48_8')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_8_unet5BM_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_9')
# dm = DataTask('tsk48_9')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_9_unetB7_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()







# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_10')
# dm = DataTask('tsk48_10')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_10_unetB8_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_11')
# dm = DataTask('tsk48_11')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_11_unetB10_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
# # # ############## Task 50   input -1 1#############
# tsm = ModelTask('tsk50_1')
# dm = DataTask('tsk50_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task50_1_unet4BNR_base_mid_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*5
#
#
# tsm.save()
# dm.save()
# run_one_task()











# # # ############## Task 51   input -1 1#############
# tsm = ModelTask('tsk51_1')
# dm = DataTask('tsk51_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task51_1_unetBS_resid_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




#
# # # ############## Task 51   input -1 1#############
# tsm = ModelTask('tsk51_2')
# dm = DataTask('tsk51_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task51_2_unetBS_resid_focal_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_0')
# dm = DataTask('tsk52_0')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_1_unetB_const_lr_update1'  #task52_0_unetB_const_lr_update1
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()

#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_1')
# dm = DataTask('tsk52_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_1_unetB_resid_const_lr_update1'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#




# # # ############## Task 51   input -1 1#############
# tsm = ModelTask('tsk51_2')
# dm = DataTask('tsk51_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_2_unetB2_resid_focal_const_lr_update1'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
#
# tsm.task_par['tsk_set']['criticUpdates'] = 1
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_3')
# dm = DataTask('tsk52_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
#
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['model_path'] ="model_best.pth.tar"
#
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_3_unetB_resid_const_lr_update3_bz_3'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
#
# tsm.task_par['tsk_set']['loss']['update_epoch'] =2
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 1
#
# tsm.task_par['tsk_set']['criticUpdates'] = 3
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_4')
# dm = DataTask('tsk52_4')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_4_unetB_resid_const_lr_update3_bz_3_loss_update6'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =4
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 3
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()






# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_5')
# dm = DataTask('tsk52_5')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_5_unet3t_base48_6_gate'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#







# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_6')
# dm = DataTask('tsk52_6')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_6_baseb11_gate'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
#
#
#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_7')
# dm = DataTask('tsk52_7')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_7_t5_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#







#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_8')
# dm = DataTask('tsk52_8')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_8_t4_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_9')
# dm = DataTask('tsk52_9')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_9_b12_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 70
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_10')
# dm = DataTask('tsk52_10')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_10_t6_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000
#
#
# tsm.save()
# dm.save()
# run_one_task()




#
#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_12')
# dm = DataTask('tsk52_12')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_12_B14_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_12_B14_base/checkpoints/epoch_100_"
# tsm.task_par['tsk_set']['dg_key_word'] = 'input3_s'
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#






#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_13')
# dm = DataTask('tsk52_13')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_13_t7_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_13_t7_base/checkpoints/epoch_150_"
# tsm.task_par['tsk_set']['dg_key_word'] = 'input3_s'
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*2
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
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_14')
# dm = DataTask('tsk52_14')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_14_b15_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data//hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_14_b15_base/checkpoints/epoch_150_"
# tsm.task_par['tsk_set']['dg_key_word'] = 'label_map'
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*2
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_15')
# dm = DataTask('tsk52_15')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_15_b16_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*2
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#





#
# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_17')
# dm = DataTask('tsk52_17')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# tsm.task_par['tsk_set']['train'] = False
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_17_b17_base/checkpoints/epoch_150_"
# tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_17_b17_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*2
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_18')
# dm = DataTask('tsk52_18')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_17_b17_base/checkpoints/epoch_150_"
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_18_t8_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # ############## Task 52   input -1 1#############
# tsm = ModelTask('tsk52_19')
# dm = DataTask('tsk52_19')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task52_17_b17_base/checkpoints/epoch_150_"
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task52_19_t9_base'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 150
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()
#


#
# # # ############## Task 68   input -1 1#############
# tsm = ModelTask('tsk68_5')
# dm = DataTask('tsk68_5')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['old_gpu_ids'] =2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/raid/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task48_5_unetB_const_lr_100/checkpoints/epoch_100_"
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task68_5_f_unet_focal'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 6
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 68   input -1 1#############
# tsm = ModelTask('tsk68_5_1')
# dm = DataTask('tsk68_5_1')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['old_gpu_ids'] =2
# tsm.task_par['tsk_set']['gpu_ids'] = 3
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task48_5_unetB_const_lr_100/checkpoints/epoch_100_"
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task68_5_1_f_unet_focal'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'focal_loss'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()

#
#
# # ############## Task 68   input -1 1#############
# tsm = ModelTask('tsk68_5_2')
# dm = DataTask('tsk68_5_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# tsm.task_par['tsk_set']['continue_train'] =True
# tsm.task_par['tsk_set']['old_gpu_ids'] =2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task48_5_unetB_const_lr_100/checkpoints/epoch_100_"
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task68_5_2_f_unet_resid_ce'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 49   input -1 1#############
# tsm = ModelTask('tsk49_2')
# dm = DataTask('tsk49_2')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task49_2_unet_t1_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
# # # ############## Task 49   input -1 1#############
# tsm = ModelTask('tsk49_3')
# dm = DataTask('tsk49_3')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task49_3_unet_t2_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_5')
# dm = DataTask('tsk48_5')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# #dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# #tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_5_unetB_const_lr_100'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['gpu_ids'] = 2
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()


# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_6')
# dm = DataTask('tsk48_6')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_6_unet3t_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()
#

#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_7')
# dm = DataTask('tsk48_7')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_7_unetbs_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()



#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_8')
# dm = DataTask('tsk48_8')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_8_unet5BM_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 3
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()




# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_9')
# dm = DataTask('tsk48_9')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_9_unetB7_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()







# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_10')
# dm = DataTask('tsk48_10')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_10_unetB8_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 1
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()


#
# # # ############## Task 48   input -1 1#############
# tsm = ModelTask('tsk48_11')
# dm = DataTask('tsk48_11')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
#
# dm.data_par['datapro']['dataset']['task_name']='hist_th_0.06'
# dm.data_par['datapro']['dataset']['prepare_data']=False
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# tsm.task_par['tsk_set']['task_name'] = 'task48_11_unetB10_const_lr'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 50
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
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
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 400*10000
#
#
# tsm.save()
# dm.save()
# run_one_task()








#
#
# # ############## Task 0   input -1 1#############
# tsm = ModelTask('task0')
# dm = DataTask('task_0')
# dm.data_par['datapro']['task_type']='seg'
# dm.data_par['datapro']['dataset']['dataset_name']='lpba'
# # dm.data_par['datapro']['dataset']['output_path']='/playpen/raid/zyshen/data/'
# # tsm.task_par['tsk_set']['save_fig_on'] = False
# # tsm.task_par['tsk_set']['train'] = False
# # tsm.task_par['tsk_set']['dg_key_word'] = 'input1_s'
# tsm.task_par['tsk_set']['continue_train'] =False
# tsm.task_par['tsk_set']['old_gpu_ids'] =2
# tsm.task_par['tsk_set']['gpu_ids'] = 0
#
# tsm.task_par['tsk_set']['model_path'] = "/playpen/zyshen/data/hist_th_0.06_lpba_seg_patchedmy_balanced_random_crop/task48_5_unetB_const_lr_100/checkpoints/epoch_100_"
# dm.data_par['datapro']['dataset']['task_name']='debug_0'
# dm.data_par['datapro']['dataset']['prepare_data']=True
# dm.data_par['datapro']['seg']['save_train_custom']=False
# dm.data_par['datapro']['seg']['num_flicker_per_train_img']=2
# dm.data_par['datapro']['seg']['patch_size']=[72,72,72]
# dm.data_par['datapro']['seg']['partition']['overlap_size']=[16,16,16]
# dm.data_par['datapro']['seg']['partition']['flicker_on']=True
# dm.data_par['datapro']['seg']['partition']['flicker_mode']='rand'
# dm.data_par['datapro']['seg']['partition']['flicker_range']=5
#
# tsm.task_par['tsk_set']['task_name'] = 'task0_debug'  #task42_unet4_base
# tsm.task_par['tsk_set']['epoch'] = 100
# tsm.task_par['tsk_set']['model'] = 'unet'
# tsm.task_par['tsk_set']['batch_sz'] = 2
# tsm.task_par['tsk_set']['val_period'] =2
# tsm.task_par['tsk_set']['loss']['update_epoch'] =-1
#
# tsm.task_par['tsk_set']['loss']['type'] = 'ce'
# tsm.task_par['tsk_set']['loss']['ce']['weighted'] = True
# tsm.task_par['tsk_set']['loss']['residue_weight_on'] = True
# tsm.task_par['tsk_set']['loss']['log_update'] = False
# tsm.task_par['tsk_set']['loss']['only_resid_update'] = True
# tsm.task_par['tsk_set']['loss']['density_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['continuous_update'] = False
# tsm.task_par['tsk_set']['loss']['residue_weight_momentum'] = 0.1
# tsm.task_par['tsk_set']['loss']['focal_loss_weight_on'] = False
# tsm.task_par['tsk_set']['loss']['activate_epoch'] = 0
#
# tsm.task_par['tsk_set']['criticUpdates'] = 2
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [400,2,1]
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/2
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
# tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*8
#
#
# tsm.save()
# dm.save()
# run_one_task()
#
#
#
