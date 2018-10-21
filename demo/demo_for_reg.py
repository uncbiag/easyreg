import matplotlib as matplt
matplt.use('Agg')

import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
#os.environ["CUDA_CACHE_PATH"] = "/playpen/zyshen/.cuda_cache
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


for sess in ['inter','intra']:
    tsm = ModelTask('task_reg')
    dm = DataTask('task_reg')
    dm.data_par['datapro']['task_type']='reg'
    dm.data_par['datapro']['dataset']['dataset_name']='oai'
    dm.data_par['datapro']['reg']['sched']= sess

    dm.data_par['datapro']['dataset']['output_path']='/playpen/zyshen/data/'
    dm.data_par['datapro']['dataset']['task_name'] = 'debugging_demo'  # 'reg_debug_labeled'#'reg_debug_2000' #'reg_debug_3000_pair' #
    """Every time prepare the data from the scratch, a data processing task name should be given,  usually the data only need to be prepared once"""
    dm.data_par['datapro']['dataset']['prepare_data'] = False
    dm.data_par['datapro']['dataset']['img_size'] = [160, 384, 384]
    dm.data_par['datapro']['reg']['input_resize_factor'] = [80. / 160., 192. / 384., 192. / 384]
    """" resize the image by [factor_x,factor_y, factor_z]"""

    tsm.task_par['tsk_set']['task_name'] = 'debugging'
    """Current task name, all tasks with different settings will use the same train/val/test data"""
    tsm.task_par['tsk_set']['train'] = False
    tsm.task_par['tsk_set']['save_by_standard_label'] = True
    tsm.task_par['tsk_set']['continue_train'] =False
    tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5
    tsm.task_par['tsk_set']['old_gpu_ids']=2
    """ the gpu id of the checkpoints to be loaded"""
    tsm.task_par['tsk_set']['gpu_ids'] = 0  #1
    """ the gpu id of the current task"""
    tsm.task_par['tsk_set']['model_path'] =''

        #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
    #                                        '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_220_',
    """the path of saved checkpoint, should always be '' unless model is 'reg_net' and need to continue the training from the checkpoints"""
    tsm.task_par['tsk_set']['input_resize_factor'] =  dm.data_par['datapro']['reg']['input_resize_factor']
    tsm.task_par['tsk_set']['img_size'] =  dm.data_par['datapro']['dataset']['img_size']
    tsm.task_par['tsk_set']['n_in_channel'] = 1
    """image channel,  for grey image should always be one"""
    tsm.task_par['tsk_set']['reg'] = {}
    """ resample the 3d image size by [factor_x, factor_y, factor_z] """
    tsm.task_par['tsk_set']['reg']['low_res_factor'] = 0.5
    """ low resolution map factor , the operations would be computed on low-resolution map, the final deformation map will be upsampled from low-resolution map"""
    tsm.task_par['tsk_set']['network_name'] ='affine'  #'mermaid' 'svf' 'syn' affine bspline
    tsm.task_par['tsk_set']['epoch'] = 300
    tsm.task_par['tsk_set']['model'] = 'mermaid_iter'  #mermaid_iter reg_net  ants  nifty_reg
    tsm.task_par['tsk_set']['batch_sz'] = 4
    tsm.task_par['tsk_set']['val_period'] =10
    """do validation every # epoch"""
    tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
    """similarity measure,  here can be mse, ncc, lncc"""
    debug_num = 4
    tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,debug_num]
    """ number of pairs for each epoch,  [200,8,5] refers to 200 pairs for each train epoch, 8 epoch for each validation epoch and 5 pairs for each debug epoch"""
    tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
    tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
    tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3
    tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5

    tsm.save()
    dm.save()
    run_one_task()