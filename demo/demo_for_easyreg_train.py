import matplotlib as matplt
matplt.use('Agg')
import SimpleITK as sitk
import sys,os
import torch
torch.backends.cudnn.benchmark=True
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
print(sys.path)
import data_pre.module_parameters as pars
import subprocess
from abc import ABCMeta, abstractmethod
from model_pool.piplines import run_one_task
from data_pre.reg_data_utils import write_list_into_txt, get_file_name



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
    def __init__(self,name,path='../settings/base_data_settings.json'):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON(path)


    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)

class ModelTask(BaseTask):
    """
    load the task settings from json
    """
    def __init__(self,name,path='../settings/base_task_settings.json'):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self,path= '../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)





def init_train_env(setting_path,data_folder, task_name, data_task_name=None):
    """
    :param task_full_path:  the path of a completed task
    :param source_path: path of the source image
    :param target_path: path of the target image
    :param l_source: path of the label of the source image
    :param l_target: path of the label of the target image
    :return: None
    """
    dm_json_path = os.path.join(setting_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_path, 'cur_task_setting.json')
    assert os.path.isfile(tsm_json_path),"task setting not exists"
    dm = DataTask('task_reg',dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg',tsm_json_path)
    data_task_name =  data_task_name if len(data_task_name) else 'custom'
    data_task_path = os.path.join(data_folder,data_task_name)
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = data_folder
        dm.data_par['datapro']['dataset']['task_name'] = data_task_name
    tsm.task_par['tsk_set']['task_name'] = task_name
    tsm.task_par['tsk_set']['data_folder'] = data_task_path

    return dm, tsm


def addition_settings(dm, tsm):
    data_task_path = tsm.task_par['tsk_set']['data_folder']
    task_name = tsm.task_par['tsk_set']['task_name']
    task_output_path = os.path.join(data_task_path, task_name)
    if args.affine_stage_in_two_stage_training:
        tsm.task_par['tsk_set']['network_name'] = 'affine_sym'
        tsm.task_par['tsk_set']['reg']['mermaid_net']['using_affine_init'] = True
        tsm.task_par['tsk_set']['reg']['mermaid_net']['affine_init_path'] = os.path.join(task_output_path,
                                                                                         'checkpoints/model_best.pth.tar')
    if args.next_stage_in_two_stage_training:
        tsm.task_par['tsk_set']['continue_train'] = False


    return dm, tsm

def backup_settings(args,dm,tsm):
    setting_folder_path = args.setting_folder_path
    task_name = args.task_name_record
    setting_backup = os.path.join(setting_folder_path, task_name)
    os.makedirs(setting_backup, exist_ok=True)
    dm_backup_json_path = os.path.join(setting_backup, 'cur_data_setting.json')
    tsm_backup_json_path =os.path.join(setting_backup,'cur_task_setting.json')
    tsm.save(tsm_backup_json_path)
    if dm is not None:
        dm.save(dm_backup_json_path)



def __do_registration_train(args,pipeline=None):

    data_folder = args.data_folder
    task_name = args.task_name
    data_task_name = args.data_task_name
    setting_folder_path = args.setting_folder_path
    data_task_path = os.path.join(data_folder,data_task_name)
    task_output_path = os.path.join(data_task_path,task_name)
    os.makedirs(task_output_path, exist_ok=True)
    dm, tsm = init_train_env(setting_folder_path,data_folder,task_name,data_task_name)
    backup_settings(args,dm, tsm)
    dm, tsm = addition_settings(dm, tsm)
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    tsm.save(tsm_json_path)
    if dm is not None:
        dm.save(dm_json_path)
    data_loaders = pipeline.data_loaders if pipeline is not None else None
    pipeline = run_one_task(tsm_json_path, dm_json_path,data_loaders)
    return pipeline

def do_registration_train(args):
    task_name = args.task_name
    args.task_name_record = task_name
    pipeline = None
    args.affine_stage_in_two_stage_training = False
    args.next_stage_in_two_stage_training = False
    if args.train_affine_first:
        args.affine_stage_in_two_stage_training = True
        args.task_name = task_name +'_stage1_affine'
        pipeline = __do_registration_train(args)
        pipeline.clean_up()
        #torch.cuda.empty_cache()
        args.affine_stage_in_two_stage_training  = False
        args.next_stage_in_two_stage_training =True
        args.setting_folder_path = os.path.join(args.setting_folder_path, task_name)
        args.task_name = task_name+'_stage2_nonp'
    __do_registration_train(args,pipeline)







if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='implementation of Adaptive vector-momentum-based Stationary Velocity Field Mappinp (AVSM)')
    parser.add_argument('-df','--data_folder', required=False, type=str,
                        default=None,help='the path of data folder')
    parser.add_argument('-dtn','--data_task_name', required=False, type=str,
                        default='',help='the name of the data related task (like subsampling)')
    parser.add_argument('-tn','--task_name', required=False, type=str,
                        default=None,help='the name of the task')
    parser.add_argument('-ts','--setting_folder_path', required=False, type=str,
                        default=None,help='path to load settings')
    parser.add_argument('--train_affine_first',required=False,action='store_true',
                        help='train affine network first, then train non-parametric network')
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')
    args = parser.parse_args()
    print(args)
    do_registration_train(args)


    # -df=/playpen/zyshen/data -dtn=croped_for_reg_debug_3000_pair_oai_reg_inter -tn=interface_vsvf -ts=/playpen/zyshen/reg_clean/demo/demo_settings/mermaid/training_network_vsvf -g=3
    # -df=/playpen/zyshen/data -dtn=croped_for_reg_debug_3000_pair_oai_reg_inter -tn=interface_vsvf -ts=/playpen/zyshen/reg_clean/demo/demo_settings/mermaid/training_network_vsvf --train_affine_first -g=2
    # -df=//playpen/zyshen/ll1/zyshen/data -dtn=croped_for_reg_debug_3000_pair_oai_reg_inter_gpu0 -tn=interface_vsvf_dev_gpu0 -ts=/playpen/zyshen/ll1/zyshen/reg_clean/demo/demo_settings/mermaid/training_network_vsvf_gpu0 --train_affine_first -g=2



    # --run_demo --demo_name=opt_vsvf -txt=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_vsvf
    # --run_demo --demo_name=opt_rdmm_predefined -txt=/playpen/zyshen/data/reg_lung_160/test/pair_path_list.txt -g=3 -o=/playpen/zyshen/data/reg_lung_160/new_interface/test_opt_rdmm_predefined
    # --run_demo --demo_name=network_vsvf -txt=/playpen/zyshen/debugs/get_val_and_debug_res/test.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_vsvf_net
    # --run_demo --demo_name=network_rdmm -txt=/playpen/zyshen/debugs/get_val_and_debug_res/test.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_rdmm_net
