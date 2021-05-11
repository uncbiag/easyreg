import os
import sys
import torch
from easyreg.piplines import run_one_task
import argparse
from task import ModelTask


class SegmentationTraining():
    def __init__(self, args):
        self.args = args


    def _set_environment(self):
        sys.path.insert(0,os.path.abspath('..'))
        sys.path.insert(0,os.path.abspath('.'))
        sys.path.insert(0,os.path.abspath('../easyreg'))
        torch.backends.cudnn.benchmark=True

    def train(self):
        return
    
    def _create_folders(self):
        self._create_folder(self.output_root_path)
        self._create_folder(self.task_output_path)
        self._create_folder(self.data_task_path)
        self._create_folder(self.setting_backup)

    
    def _create_folder(self, path):

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print("Warning, {} exists, press Y/y to overide, N/n to stop")
            user_inp = input()
            if user_inp in ["Y", "y"]:
                os.makedirs(path)
                
            
            elif user_inp in ["N", "n"]:
                exit()

    def __do_segmentation_train(self, pipeline=None):
    """
        set running env and run the task
        :param pipeline:a Pipeline object
        :return: a Pipeline object
    """
    self.pipeline = pipeline
    self.output_root_path = self.args.output_root_path
    self.task_name = self.args.task_name
    self.data_task_name = self.args.data_task_name
    self.setting_folder_path = self.args.setting_folder_path
    self.data_task_path = os.path.join(output_root_path,data_task_name)
    self.task_output_path = os.path.join(data_task_path,task_name)
    os.makedirs(task_output_path, exist_ok=True)

    dm, tsm = self.init_train_env()
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    self.dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    self.tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    tsm.save(self.tsm_json_path)
    if dm is not None:
        dm.save(self.dm_json_path)
    data_loaders = pipeline.data_loaders if self.pipeline is not None else None
    self.pipeline = run_one_task(self.tsm_json_path, self.dm_json_path, data_loaders)

def init_train_env(self):

    assert os.path.isfile(self.tsm_json_path),"task setting not exists"

    dm = DataTask('task_reg', self.dm_json_path) if os.path.isfile(self.dm_json_path) else None
    tsm = ModelTask('task_reg',tsm_json_path)
    self.data_task_name = self.data_task_name if len(self.data_task_name)>0 else 'custom'
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = self.output_root_path
        dm.data_par['datapro']['dataset']['task_name'] = self.data_task_name
    tsm.task_par['tsk_set']['task_name'] = self.task_name
    tsm.task_par['tsk_set']['output_root_path'] = self.data_task_path
    return dm, tsm

def save_settings(self):
    self.setting_folder_path = args.setting_folder_path
    self.dm_json_path = os.path.join(setting_folder_path, 'cur_data_setting.json')
    self.tsm_json_path = os.path.join(setting_folder_path, 'cur_task_setting.json')
    dm = DataTask('task_reg', self.dm_json_path) if os.path.isfile(self.dm_json_path) else None
    tsm = ModelTask('task_reg', tsm_json_path)
    task_name = args.task_name_record
    setting_backup = os.path.join(setting_folder_path, task_name+'_backup')
    os.makedirs(setting_backup, exist_ok=True)
    dm_backup_json_path = os.path.join(setting_backup, 'cur_data_setting.json')
    tsm_backup_json_path =os.path.join(setting_backup,'cur_task_setting.json')
    tsm.save(tsm_backup_json_path)
    if dm is not None:
        dm.save(dm_backup_json_path)






if __name__ == '__main__':
    """
        An interface for learning segmentation methods.
        This script will generate the three folders for the training, if the folder is not found in the given path
        It is recommended to use CUDA_VISIBLE_DEVICES to control the data parallelism, but it is possible to 
        Assume there is three level folder, output_root_path/ data_task_folder/ task_folder 
        Arguments: 
            --output_root_path/ -o: the path of output folder
            --data_task_name/ -dtn: data task name i.e. lung_reg_task , oai_reg_task
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --setting_folder_path/ -ts: path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: gpu_id to use
    """

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
    trainer = SegmentationTraining(args)
    trainer.train()
    # do_segmentation_train(args)

