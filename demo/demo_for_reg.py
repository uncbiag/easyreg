import matplotlib as matplt
matplt.use('Agg')

import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
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
    """
    load the data settings from json
    """
    def __init__(self,name):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON('../settings/base_data_settings.json')


    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)

class ModelTask(BaseTask):
    """
    load the task settings from json
    """
    def __init__(self,name):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON('../settings/base_task_settings.json')

    def save(self,path= '../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)



"""
The code is separated into two part,  data_preprocessing,  run_task.

The output is organized by root_path/data_task_name/cur_task_name
A notation: We provided the data_preprocessing code for oai dataset. For other dataset, please follow the guideline
"""


tsm = ModelTask('task_reg')
dm = DataTask('task_reg')
root_path ='/playpen/zyshen/data/'
data_task_name ='debugging_demo'
cur_task_name = 'debugging'

dm.data_par['datapro']['task_type']='reg'
""" task type,  only support 'reg' """
dm.data_par['datapro']['dataset']['dataset_name']='oai'
""" dataset name,  is not used if the data has already been manually prepared"""
dm.data_par['datapro']['reg']['sched']= 'intra'
""" dataset type, 'intra' for longitudinal,'inter' for cross-subject,  is not used if the data has manually prepared"""
dm.data_par['datapro']['dataset']['output_path']= root_path
""" the root path, refers to the [root_path]  in  root_path/data_task_name/cur_task_name  """
dm.data_par['datapro']['dataset']['task_name'] = data_task_name
""" the data_task_name, refers to the [data_task_name]  in  root_path/data_task_name/cur_task_name  """
dm.data_par['datapro']['dataset']['prepare_data'] = False
""" prepare the data from the scratch,  usually the data only need to be prepared once"""
dm.data_par['datapro']['dataset']['img_size'] =[160,384,384] #[193, 193, 229]
""" the numpy coordinate of the image size"""
dm.data_par['datapro']['reg']['input_resize_factor'] =[80./160.,192./384.,192./384] #[96. / 193., 96. / 193, 112. / 229.]
""" resize the image by [factor_x,factor_y, factor_z]"""
dm.data_par['datapro']['reg']['max_pair_for_loading'] = [100,10,30,10]
""" limit the max number of the pairs for [train, val, test, debug]"""
dm.data_par['datapro']['reg']['load_training_data_into_memory'] = False
""" load all training pairs into memory"""



tsm.task_par['tsk_set']['task_name'] = cur_task_name
""" the cur_task_name, refers to the [cur_task_name]  in  root_path/data_task_name/cur_task_name  """
tsm.task_par['tsk_set']['train'] = False
""" train the model """
tsm.task_par['tsk_set']['save_by_standard_label'] = True
""" save the label in original label index, for example if the original label is a way like [ 1,3,7,8], otherwise save in [0,1,2,3] """
tsm.task_par['tsk_set']['continue_train'] =False
""" train from the checkpoint"""
tsm.task_par['tsk_set']['continue_train_lr'] = 5e-5
""" set the learning rate when continue the train"""
tsm.task_par['tsk_set']['old_gpu_ids']=2
""" no longer used"""
tsm.task_par['tsk_set']['gpu_ids'] = 2
""" the gpu id of the current task"""
tsm.task_par['tsk_set']['model_path'] = '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_170_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_affine_net_sym_lncc/checkpoints/epoch_1070_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
"""the path of saved checkpoint, for methods not using network, leave it to be '' """
tsm.task_par['tsk_set']['input_resize_factor'] =  dm.data_par['datapro']['reg']['input_resize_factor']
tsm.task_par['tsk_set']['img_size'] =  dm.data_par['datapro']['dataset']['img_size']
tsm.task_par['tsk_set']['n_in_channel'] = 1
"""image channel,  for grey image should always be one"""


"""
general settings for tasks
models support list: 'reg_net'  'mermaid_iter'  'ants'  'nifty_reg' 'demons'
each model supports several methods
methods support by reg_net: 'affine_sim','affine_cycle','affine_sym', 'mermaid'
methods support by mermaid_iter: 'affine','svf' ( including affine registration first)
methods support by ants: 'affine','syn' ( including affine registration first)
methods support by nifty_reg: 'affine','bspline' ( including affine registration first)
methods support by demons: 'demons' ( including niftyreg affine registration first)


reg_net refers to registration network. 
affine_sim refers to single affine network, 
affine_cycle refers to multi-step affine network,
affine_sym refers to multi-step affine symmetric network (s-t, t-s),
mermaid refers to the mermaid library ( including various fluid based registration methods, our implementation is based on velocity momentum based svf method)
mermaid_iter refers to mermaid library, here we compare with 'affine' and 'svf' ( actually the velocity momentum based svf) method
ant refers to AntsPy: https://github.com/ANTsX/ANTsPy
nifty_reg refers : http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg
demons refers to deformably register two images using a symmetric forces demons algorithm, which is provided by simple itk

"""


tsm.task_par['tsk_set']['reg'] = {}
""" settings for registration task"""
tsm.task_par['tsk_set']['reg']['low_res_factor'] = 0.5
""" low resolution map factor for vSVF method, the operations would be computed on low-resolution map"""
tsm.task_par['tsk_set']['network_name'] ='mermaid'  #'mermaid' 'svf' 'syn' affine bspline
""" see guideline"""
tsm.task_par['tsk_set']['epoch'] = 300
""" number of training epoch"""
tsm.task_par['tsk_set']['model'] = 'reg_net'  #mermaid_iter reg_net  ants  nifty_reg
""" support  'reg_net'  'mermaid_iter'  'ants'  'nifty_reg' 'demons' """
tsm.task_par['tsk_set']['batch_sz'] = 1
""" batch size"""
tsm.task_par['tsk_set']['val_period'] =10
""" do validation every # epoch"""
tsm.task_par['tsk_set']['loss']['type'] = 'lncc'
"""similarity measure, mse, ncc, lncc"""
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,8,4]
""" number of pairs per training/val/debug epoch,  [200,8,5] refers to 200 pairs for each train epoch, 8 pairs for each validation epoch and 5 pairs for each debug epoch"""
tsm.task_par['tsk_set']['optim']['lr'] = 5e-5
"""the learning rate"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
""" 'custom','plateau',  learning rate scheduler, 'plateau' is not fully tested"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3
"""decay period of the learning rate"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5
"""decay factor of the learning rate"""


tsm.task_par['tsk_set']['reg']['mermaid_net']={}
""" settings for mermaid net"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_index_coord']=False
""" using index coordinate"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_sym']=False
""" using symmetric training, if True, the loss is combined with source-target loss, target-source loss, and symmetric loss"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['sym_factor']=100
""" the weight for symmetric factor"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_complex_net']=True
""" True : using a deep residual unet, False: using a simple unet"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_multi_step']=True
""" using multi-step training for mermaid_based method"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['num_step']=2
""" number of steps in multi-step mermaid_based method training"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_lddmm']=False
""" True: using lddmm False: using vSVF"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_affine_init']=True
""" True: using affine_network for initialized, which should be trained first. False: using id transform as initialization"""
affine_path = '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_affine_net_sym_lncc/checkpoints/epoch_1070_'
tsm.task_par['tsk_set']['reg']['mermaid_net']['affine_init_path']=affine_path
""" if using_affine_init = True, the provide the path of affine_model"""

tsm.task_par['tsk_set']['reg']['affine_net']={}
""" settings for multi-step affine network"""
tsm.task_par['tsk_set']['reg']['affine_net']['affine_net_iter']=6
""" number of steps used """
tsm.task_par['tsk_set']['reg']['affine_net']['using_complex_net']=True
""" using complex version of affine network"""


tsm.task_par['tsk_set']['reg']['mermaid_iter']={}
""" settings for the mermaid optimization version, we only provide parameter that may different in longitudinal and cross subject task"""
tsm.task_par['tsk_set']['reg']['mermaid_iter']['affine']={}
""" settings for the mermaid-affine optimization version"""
tsm.task_par['tsk_set']['reg']['mermaid_iter']['affine']['sigma']=0.7  # recommand np.sqrt(batch_sz/4) for longitudinal, recommand np.sqrt(batch_sz/2) for cross-subject


task_full_path = os.path.join(os.path.join(root_path,data_task_name), cur_task_name)
data_json_path = os.path.join(task_full_path,'cur_data_setting.json')
tsk_json_path = os.path.join(task_full_path,'cur_task_setting.json')
tsm.save(tsk_json_path)
dm.save(data_json_path)
tsm.save()
dm.save()
run_one_task()