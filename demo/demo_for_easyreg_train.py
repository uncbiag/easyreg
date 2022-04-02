import matplotlib as matplt
matplt.use('Agg')
import os, sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('../easy_reg'))
#sys.path.insert(0,os.path.abspath('../mermaid'))
import numpy as np
import torch
import random
torch.backends.cudnn.benchmark=True
import tools.module_parameters as pars
from abc import ABCMeta, abstractmethod
from easyreg.piplines import run_one_task



class BaseTask():
    __metaclass__ = ABCMeta
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def save(self):
        pass

class DataTask(BaseTask):
    """
    base module for data setting files (.json)
    """
    def __init__(self,name,path='../settings/base_data_settings.json'):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON(path)


    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)

class ModelTask(BaseTask):
    """
    base module for task setting files (.json)
    """
    def __init__(self,name,path='../settings/base_task_settings.json'):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self,path= '../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)





def init_train_env(setting_path,output_root_path, task_name, data_task_name=None):
    """
    create train environment.

    :param setting_path: the path to load 'cur_task_setting.json' and 'cur_data_setting.json' (optional if the related settings are in cur_task_setting)
    :param output_root_path: the output path
    :param data_task_name: data task name i.e. lung_reg_task , oai_reg_task
    :param task_name: task name i.e. run_training_vsvf_task, run_training_rdmm_task
    :return:
    """
    dm_json_path = os.path.join(setting_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_path, 'cur_task_setting.json')
    assert os.path.isfile(tsm_json_path),"task setting not exists"
    dm = DataTask('task_reg',dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg',tsm_json_path)
    data_task_name =  data_task_name if len(data_task_name) else 'custom'
    data_task_path = os.path.join(output_root_path,data_task_name)
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = output_root_path
        dm.data_par['datapro']['dataset']['task_name'] = data_task_name
    tsm.task_par['tsk_set']['task_name'] = task_name
    tsm.task_par['tsk_set']['output_root_path'] = data_task_path
    if tsm.task_par['tsk_set']['model']=='reg_net'and 'mermaid' in tsm.task_par['tsk_set']['method_name']:
        mermaid_setting_json = tsm.task_par['tsk_set']['reg']['mermaid_net']['mermaid_net_json_pth']
        if len(mermaid_setting_json) == 0:
            tsm.task_par['tsk_set']['reg']['mermaid_net']['mermaid_net_json_pth'] = os.path.join(setting_path,'mermaid_nonp_settings.json')

    return dm, tsm


def addition_settings_for_two_stage_training(dm, tsm):
    """
    addition settings when perform two-stage training, we assume the affine is the first stage, a non-linear method is the second stage

    :param dm: ParameterDict, data processing setting (not used for now)
    :param tsm: ParameterDict, task setting
    :return: tuple of ParameterDict,  datapro (optional) and tsk_set
    """

    if args.affine_stage_in_two_stage_training:
        tsm.task_par['tsk_set']['method_name'] = 'affine_sym'

    if args.next_stage_in_two_stage_training:
        data_task_path = tsm.task_par['tsk_set']['output_root_path']
        task_name = tsm.task_par['tsk_set']['task_name'].replace('_stage2_nonp','_stage1_affine')
        task_output_path = os.path.join(data_task_path, task_name)
        tsm.task_par['tsk_set']['continue_train'] = False
        tsm.task_par['tsk_set']['reg']['mermaid_net']['using_affine_init'] = True
        tsm.task_par['tsk_set']['reg']['mermaid_net']['affine_init_path'] = os.path.join(task_output_path,
                                                                                         'checkpoints/model_best.pth.tar')


    return dm, tsm

def backup_settings(args):
    """
    The settings saved in setting_folder_path/task_name/cur_data_setting.json and setting_folder_path/task_name/cur_task_setting.json

    :param args:
    :return: None
    """
    setting_folder_path = args.setting_folder_path
    dm_json_path = os.path.join(setting_folder_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_folder_path, 'cur_task_setting.json')
    dm = DataTask('task_reg', dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg', tsm_json_path)
    task_name = args.task_name_record
    setting_backup = os.path.join(setting_folder_path, task_name+'_backup')
    os.makedirs(setting_backup, exist_ok=True)
    dm_backup_json_path = os.path.join(setting_backup, 'cur_data_setting.json')
    tsm_backup_json_path =os.path.join(setting_backup,'cur_task_setting.json')
    if tsm.task_par['tsk_set']['model']=='reg_net' and 'mermaid' in tsm.task_par['tsk_set']['method_name']:
        mermaid_backup_json_path = os.path.join(setting_backup, 'mermaid_nonp_settings.json')
        mermaid_setting_json = tsm.task_par['tsk_set']['reg']['mermaid_net']['mermaid_net_json_pth']
        if len(mermaid_setting_json)==0:
            mermaid_setting_json = os.path.join(setting_folder_path,'mermaid_nonp_settings.json')
        mermaid_setting =pars.ParameterDict()
        mermaid_setting.load_JSON(mermaid_setting_json)
        mermaid_setting.write_ext_JSON(mermaid_backup_json_path)
    tsm.save(tsm_backup_json_path)
    if dm is not None:
        dm.save(dm_backup_json_path)




def __do_registration_train(args,pipeline=None):
    """
        set running env and run the task

        :param args: the parsed arguments
        :param pipeline:a Pipeline object, only used for two-stage training, the pipeline of the first stage (including dataloader) would be pass to the second stage
        :return: a Pipeline object
    """

    output_root_path = args.output_root_path
    task_name = args.task_name
    data_task_name = args.data_task_name
    setting_folder_path = args.setting_folder_path
    data_task_path = os.path.join(output_root_path,data_task_name)
    task_output_path = os.path.join(data_task_path,task_name)
    os.makedirs(task_output_path, exist_ok=True)
    dm, tsm = init_train_env(setting_folder_path,output_root_path,task_name,data_task_name)
    dm, tsm = addition_settings_for_two_stage_training(dm, tsm)
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    tsm.save(tsm_json_path)
    if dm is not None:
        dm.save(dm_json_path)
    data_loaders = pipeline.data_loaders if pipeline is not None else None
    pipeline = run_one_task(tsm_json_path, dm_json_path,data_loaders)
    return pipeline


def set_seed_for_demo(args):
    """ reproduce the training demo"""
    seed = 2018
    if args.is_demo:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True




def do_registration_train(args):
    """
    a interface for setting one-stage training or two stage training (include affine)

    :param args: the parsed arguments
    :return: None
    """
    set_seed_for_demo(args)
    task_name = args.task_name
    args.task_name_record = task_name
    backup_settings(args)
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
        args.setting_folder_path = os.path.join(args.setting_folder_path, task_name+'_backup')
        args.task_name = task_name+'_stage2_nonp'
    __do_registration_train(args,pipeline)







if __name__ == '__main__':
    """
        A training interface for learning methods.
        The method support list :  mermaid-related methods (vSVF,LDDMM,RDMM), voxel-morph (cvpr and miccai)
        Assume there is three level folder, output_root_path/ data_task_name/ task_name 
        In data_task_folder, you must include train/val/test/debug folders, for details please refer to doc/source/notes/preapre_data.rst
        Arguments: 
            --output_root_path/ -o: the path of easyreg output root folder
            --data_task_name/ -dtn: data task name i.e. lung_reg_task , oai_reg_task,
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)
            --train_affine_first: train affine network first, then train non-parametric network
            --gpu_id/ -g: gpu_id to use
            --is_demo: reproduce the tutorial result
    """
    import argparse

    parser = argparse.ArgumentParser(description="An easy interface for training registration models")
    parser.add_argument('-o','--output_root_path', required=False, type=str,
                        default=None,help='the path of output folder')
    parser.add_argument('-dtn','--data_task_name', required=False, type=str,
                        default='',help='the name of the data related task (like subsampling)')
    parser.add_argument('-tn','--task_name', required=False, type=str,
                        default=None,help='the name of the task')
    parser.add_argument('-ts','--setting_folder_path', required=False, type=str,
                        default=None,help='path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)')
    parser.add_argument('--train_affine_first',required=False,action='store_true',
                        help='train affine network first, then train non-parametric network')
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')
    parser.add_argument('--is_demo', required=False,action='store_true', help="reproduce the tutorial result")
    args = parser.parse_args()
    print(args)
    do_registration_train(args)

