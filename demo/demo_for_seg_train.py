import matplotlib as matplt
matplt.use('Agg')
import os, sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('../easyreg'))
import numpy as np
import torch
import random
import tools.module_parameters as pars
from abc import ABCMeta, abstractmethod
from easyreg.piplines import run_one_task
torch.backends.cudnn.benchmark=True



class BaseTask():
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def save(self):
        pass


class DataTask(BaseTask):
    """
    base module for data setting files (.json)
    """

    def __init__(self, name, path='../settings/base_data_settings.json'):
        super(DataTask, self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON(path)

    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)


class ModelTask(BaseTask):
    """
    base module for task setting files (.json)
    """

    def __init__(self, name, path='../settings/base_task_settings.json'):
        super(ModelTask, self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self, path='../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)




def init_train_env(setting_path,output_root_path, task_name, data_task_name=None):
    """
    create train environment.

    :param setting_path: the path to load 'cur_task_setting.json' and 'cur_data_setting.json' (optional if the related settings are in cur_task_setting)
    :param output_root_path: the output path
    :param data_task_name: data task name i.e. lung_seg_task , oai_seg_task
    :param task_name: task name i.e. run_unet, run_with_ncc_loss
    :return:
    """
    dm_json_path = os.path.join(setting_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_path, 'cur_task_setting.json')
    assert os.path.isfile(tsm_json_path),"task setting {} not exists".format(tsm_json_path)
    dm = DataTask('task_reg',dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg',tsm_json_path)
    data_task_name =  data_task_name if len(data_task_name) else 'custom'
    data_task_path = os.path.join(output_root_path,data_task_name)
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = output_root_path
        dm.data_par['datapro']['dataset']['task_name'] = data_task_name
    tsm.task_par['tsk_set']['task_name'] = task_name
    tsm.task_par['tsk_set']['output_root_path'] = data_task_path
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
    tsm.save(tsm_backup_json_path)
    if dm is not None:
        dm.save(dm_backup_json_path)




def __do_segmentation_train(args,pipeline=None):
    """
        set running env and run the task

        :param args: the parsed arguments
        :param pipeline:a Pipeline object
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
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    tsm.save(tsm_json_path)
    if dm is not None:
        dm.save(dm_json_path)
    data_loaders = pipeline.data_loaders if pipeline is not None else None
    pipeline = run_one_task(tsm_json_path, dm_json_path,data_loaders)
    return pipeline




def do_segmentation_train(args):
    """

    :param args: the parsed arguments
    :return: None
    """
    task_name = args.task_name
    args.task_name_record = task_name
    backup_settings(args)
    pipeline = None
    __do_segmentation_train(args,pipeline)







if __name__ == '__main__':
    """
        An interface for learning segmentation methods.
        Assume there is three level folder, output_root_path/ data_task_name/ task_name
        In data_task_folder, you must include train/val/test/debug folders, for details please refer to doc/source/notes/preapre_data.rst
        Arguments: 
            --output_root_path/ -o: the path of output folder
            --data_task_name/ -dtn: data task name i.e. lung_reg_task , oai_reg_task
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(description="An easy interface for training segmentation models")
    parser.add_argument('-o','--output_root_path', required=False, type=str,
                        default=None,help='the path of output folder')
    parser.add_argument('-dtn','--data_task_name', required=False, type=str,
                        default='',help='the name of the data related task (like subsampling)')
    parser.add_argument('-tn','--task_name', required=False, type=str,
                        default=None,help='the name of the task')
    parser.add_argument('-ts','--setting_folder_path', required=False, type=str,
                        default=None,help='path of the folder where settings are saved,should include cur_task_setting.json)')
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')
    args = parser.parse_args()
    print(args)
    do_segmentation_train(args)

