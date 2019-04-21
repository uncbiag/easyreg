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

import argparse

parser = argparse.ArgumentParser(description='Registeration demo (include train and test)')

parser.add_argument('--gpu', required=False, type=int, default=2,
                    help='give the id to run the gpu')
parser.add_argument('--llf', required=False, type=bool, default=False,
                    help='run on long leaf')
args = parser.parse_args() 

tsm = ModelTask('task_reg')
dm = DataTask('task_reg')
if not args.llf:
    root_path = '/playpen/zyshen/data/'
    data_task_name ='croped_for_reg_debug_3000_pair_oai_reg_inter' #'reg_debug_3000_pair_oasis3_reg_inter'#
    data_task_name_for_train ='croped_for_reg_debug_3000_pair_oai_reg_inter' #'reg_debug_3000_pair_oasis3_reg_inter'#
else:
    root_path = '/pine/scr/z/y/zyshen/expri/'
    data_task_name = 'croped_for_reg_debug_3000_pair_oai_reg_inter'  # 'reg_debug_3000_pair_oasis3_reg_inter'#
    data_task_name_for_train = 'croped_for_reg_debug_3000_pair_oai_reg_inter'  # 'reg_debug_3000_pair_oasis3_reg_inter'#

cur_program_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_output_path = os.path.join(root_path,data_task_name_for_train)
cur_task_name ='todel'#'reg_adpt_lddamm_wkw_formul_4_1_omt_2step_1000sym'#reg_adpt_lddmm_05fix_omt4_2degree_ls_004_2step_1000sym_onestep'#reg_adpt_lddmm_new_2step_ls01_1000sym_onestep_2bz' # vm_cvprwithregfix5000'
#'reg_fixed_lddmm_onestepphi_reg3_sunet_clamp_omt_IT_net_001rloss1_withinit'
#'reg_fixed_lddmm_onestepphi_reg3_unet_clamp_sym500_omt_001rloss1_fixed_continue' # vm_cvprwithregfix5000'
is_oai = 'oai' in data_task_name
is_oasis = not is_oai
img_sz = [160/2,384/2,384/2]
input_resize_factor = [1.,1.,1.]
spacing = [1. / (img_sz[i] * input_resize_factor[i]-1) for i in range(len(img_sz))]

# if is_oai:
#     img_sz = [160,384,384]
#     input_resize_factor = [80./160.,192./384.,192./384]
#     spacing = [1. / (img_sz[i] * input_resize_factor[i]-1) for i in range(len(img_sz))]
# if is_oasis:
#     img_sz = [224,224,224]
#     input_resize_factor = [128./224,128./224,128./224]
#     spacing = [1. / (img_sz[i] * input_resize_factor[i]-1) for i in range(len(img_sz))]


dm.data_par['datapro']['task_type']='reg'
""" task type,  only support 'reg' """
dm.data_par['datapro']['dataset']['dataset_name']='oai'
""" dataset name,  is not used if the data has already been manually prepared"""
dm.data_par['datapro']['reg']['sched']= 'inter'  # no usage now
""" dataset type, 'intra' for longitudinal,'inter' for cross-subject,  is not used if the data has manually prepared"""
dm.data_par['datapro']['dataset']['output_path']= root_path
""" the root path, refers to the [root_path]  in  root_path/data_task_name/cur_task_name  """
dm.data_par['datapro']['dataset']['task_name'] = data_task_name
""" the data_task_name, refers to the [data_task_name]  in  root_path/data_task_name/cur_task_name  """
dm.data_par['datapro']['dataset']['prepare_data'] = False
""" prepare the data from the scratch,  usually the data only need to be prepared once"""
dm.data_par['datapro']['dataset']['img_size'] =img_sz#[160,384,384] #[193, 193, 229]
""" the numpy coordinate of the image size"""
dm.data_par['datapro']['reg']['input_resize_factor'] =input_resize_factor#[96. / 193., 96. / 193, 112. / 229.]
""" resize the image by [factor_x,factor_y, factor_z]"""
dm.data_par['datapro']['dataset']['spacing'] =spacing#[96. / 193., 96. / 193, 112. / 229.]
""" spacing of image """
dm.data_par['datapro']['reg']['max_pair_for_loading'] = [-1,-1,-1,-1]
""" limit the max number of the pairs for [train, val, test, debug]"""
dm.data_par['datapro']['reg']['load_training_data_into_memory'] = False
""" load all training pairs into memory"""



tsm.task_par['tsk_set']['task_name'] = cur_task_name
""" the cur_task_name, refers to the [cur_task_name]  in  root_path/data_task_name/cur_task_name  """
tsm.task_par['tsk_set']['train'] = True
""" train the model """
tsm.task_par['tsk_set']['save_by_standard_label'] = True
""" save the label in original label index, for example if the original label is a way like [ 1,3,7,8], otherwise save in [0,1,2,3] """
tsm.task_par['tsk_set']['continue_train'] =True
""" train from the checkpoint"""
tsm.task_par['tsk_set']['load_model_but_train_from_begin'] =True ###############TODO  should be false
tsm.task_par['tsk_set']['load_model_but_train_from_epoch'] =50 ###############TODO  should be false

""" load the saved model as initialization, but still will train the whole model from the beginning"""
tsm.task_par['tsk_set']['continue_train_lr'] = 2e-5/2   #  TODO to be put back to 5e-5
""" set the learning rate when continue the train"""
tsm.task_par['tsk_set']['old_gpu_ids']=0
""" no longer used"""
tsm.task_par['tsk_set']['gpu_ids'] = args.gpu
""" the gpu id of the current task"""
tsm.task_par['tsk_set']['model_path'] =os.path.join(data_output_path,'reg_adpt_lddmm_05fix_omt4_2degree_ls_004_2step_1000sym_onestep/checkpoints/epoch_100_')
    #"/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_110_"

    #'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/reg_fixed_lddmm_onestepphi_reg3_unet_clamp_sym500_omt_001rloss1_fixed_continue/checkpoints/epoch_40_'

    #"/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_110_"

    #'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/reg_svf_baseline/checkpoints/epoch_70_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debug_on_lddmm_network_bz2/checkpoints/epoch_300_'

    #'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/reg_fixed_lddmm_onestepphi_reg3_unet_clamp_sym500_omt_001rloss1_fixed_continue/checkpoints/epoch_40_'
    #'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/reg_fixed_lddmm_onestepphi_reg3_unet_clamp_sym500_omt001rloss1_fixed/checkpoints/epoch_30_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debug_reg_lddmm_unet_oneadpt_5scale_noloss_10reg_softmax/checkpoints/epoch_90_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debug_reg_lddmm_unet_oneadpt_5scale_noloss_3reg_softmax_continue5iter_withclamp/checkpoints/epoch_100_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/reg_fixed_lddmm_onestepphi_reg3_sunet_clamp/checkpoints/epoch_5_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debug_reg_lddmm_unet_oneadpt_5scale_noloss_10reg_softmax/checkpoints/epoch_90_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debug_on_lddmm_network_bz2/checkpoints/epoch_300_'
#'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/vm_cvprwithregfixnccandtraindata10000_withaffine/checkpoints/epoch_60_'
#"/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_110_"
    #"/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/debugging_smoother_clamp_withunet_noclamp/checkpoints/epoch_120_"
    #''/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/checkpoints/epoch_170_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_affine_net_sym_lncc/checkpoints/epoch_1070_'
    #'/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_inter/train_inter_mermaid_net_reisd_2step_lncc_lgreg10_sym_recbi/checkpoints/epoch_100_'
"""the path of saved checkpoint, for methods not using network, leave it to be '' """
tsm.task_par['tsk_set']['input_resize_factor'] =  dm.data_par['datapro']['reg']['input_resize_factor']
tsm.task_par['tsk_set']['img_size'] =  dm.data_par['datapro']['dataset']['img_size']
tsm.task_par['tsk_set']['spacing'] =  dm.data_par['datapro']['dataset']['spacing']
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



tsm.task_par['tsk_set']['network_name'] ='mermaid'  #'mermaid' 'svf' 'syn' affine bspline
""" see guideline"""
tsm.task_par['tsk_set']['epoch'] = 50
""" number of training epoch"""
tsm.task_par['tsk_set']['model'] = 'reg_net'  #mermaid_iter reg_net  ants  nifty_reg
""" support  'reg_net'  'mermaid_iter'  'ants'  'nifty_reg' 'demons' """
tsm.task_par['tsk_set']['batch_sz'] = 1
""" batch size"""
tsm.task_par['tsk_set']['val_period'] =10
""" do validation every # epoch"""
tsm.task_par['tsk_set']['loss']['type'] = 'lncc' #######################TODO  here  should be lncc
"""similarity measure, mse, ncc, lncc"""
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [20,0,0] #[200,8,4]
""" number of pairs per training/val/debug epoch,  [200,8,5] refers to 200 pairs for each train epoch, 8 pairs for each validation epoch and 5 pairs for each debug epoch"""
tsm.task_par['tsk_set']['optim']['lr'] = 1e-4/ tsm.task_par['tsk_set']['batch_sz']  ############TODO  1e-4
"""the learning rate"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['type'] = 'custom'
""" 'custom','plateau',  learning rate scheduler, 'plateau' is not fully tested"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['step_size'] = 4000*3
"""decay period of the learning rate"""
tsm.task_par['tsk_set']['optim']['lr_scheduler']['custom']['gamma'] = 0.5
"""decay factor of the learning rate"""



tsm.task_par['tsk_set']['reg'] = {}
""" settings for registration task"""
tsm.task_par['tsk_set']['reg']['low_res_factor'] = 0.5
""" low resolution map factor for vSVF method, the operations would be computed on low-resolution map"""
tsm.task_par['tsk_set']['reg']['mermaid_net']={}
""" settings for mermaid net"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_physical_coord']=False
""" using index coordinate"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['optimize_momentum_network'] = True and tsm.task_par['tsk_set']['train']
fix_ind = [list(range(i*5,(i+1)*5)) for i in range(0,20,2)]
fix_ind_com = [list(range((i+1)*5,(i+2)*5)) for i in range(0,20,2)]
from functools import reduce
fix_ind = reduce(lambda x,y: x+y,fix_ind)
fix_ind_com = reduce(lambda x,y: x+y,fix_ind_com)
tsm.task_par['tsk_set']['reg']['mermaid_net']['epoch_list_fixed_momentum_network'] =[-1]
tsm.task_par['tsk_set']['reg']['mermaid_net']['epoch_list_fixed_deep_smoother_network'] =[-1] #fix_ind_com
tsm.task_par['tsk_set']['reg']['mermaid_net']['clamp_momentum'] =True  # TODO it should be False in most case




tsm.task_par['tsk_set']['reg']['mermaid_net']['using_sym']=True
""" using symmetric training, if True, the loss is combined with source-target loss, target-source loss, and symmetric loss"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['sym_factor']=1000
""" the weight for symmetric factor"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_complex_net']=True
""" True : using a deep residual unet, False: using a simple unet"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['num_step']=2
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_multi_step']=tsm.task_par['tsk_set']['reg']['mermaid_net']['num_step']>1
""" using multi-step training for mermaid_based method"""
""" number of steps in multi-step mermaid_based method training"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['mermaid_net_json_pth']=os.path.join(cur_program_path,'mermaid_settings/cur_settings_adpt_lddmm_new_fix_wkw.json')
""" True: using lddmm False: using vSVF"""
tsm.task_par['tsk_set']['reg']['mermaid_net']['using_affine_init']=True  # this should be true
tsm.task_par['tsk_set']['reg']['mermaid_net']['load_trained_affine_net']=True
""" True: using affine_network for initialized, which should be trained first. False: using id transform as initialization"""
affine_path = os.path.join(data_output_path, 'train_affine_net_sym_lncc/checkpoints/epoch_1070_')
tsm.task_par['tsk_set']['reg']['mermaid_net']['affine_init_path']=affine_path
""" if using_affine_init = True, the provide the path of affine_model"""

tsm.task_par['tsk_set']['reg']['affine_net']={}
""" settings for multi-step affine network"""
tsm.task_par['tsk_set']['reg']['affine_net']['affine_net_iter']=7
""" number of steps used """
tsm.task_par['tsk_set']['reg']['affine_net']['using_complex_net']=True
""" using complex version of affine network"""


tsm.task_par['tsk_set']['reg']['mermaid_iter']={}
""" settings for the mermaid optimization version, we only provide parameter that may different in longitudinal and cross subject task"""
tsm.task_par['tsk_set']['reg']['mermaid_iter']['affine']={}
""" settings for the mermaid-affine optimization version"""
tsm.task_par['tsk_set']['reg']['mermaid_iter']['affine']['sigma']=0.7  # recommand np.sqrt(batch_sz/4) for longitudinal, recommand np.sqrt(batch_sz/2) for cross-subject


task_full_path = os.path.join(os.path.join(root_path,data_task_name), cur_task_name)
os.makedirs(task_full_path,exist_ok=True)
data_json_path = os.path.join(task_full_path,'cur_data_setting.json')
tsk_json_path = os.path.join(task_full_path,'cur_task_setting.json')
tsm.save(tsk_json_path)
dm.save(data_json_path)
tsm.save()
dm.save()
run_one_task()