"""
A demo for fluid-based data augmentation.
"""

import matplotlib as matplt
import subprocess
matplt.use('Agg')
import os, sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../easy_reg'))
# sys.path.insert(0,os.path.abspath('../mermaid'))
import tools.module_parameters as pars
from abc import ABCMeta, abstractmethod
from easyreg.aug_utils import *




def init_aug_env(args):
    task_output_path = args.task_output_path
    os.makedirs(task_output_path, exist_ok=True)
    run_demo = args.run_demo
    if run_demo:
        demo_name = args.demo_name
        setting_folder_path = os.path.join('./demo_settings/data_aug', demo_name)
        output_pair_list_txt = os.path.join(os.path.join('./demo_data_aug', demo_name),"pair_to_reg.txt")

    else:
        setting_folder_path = args.setting_folder_path
        output_pair_list_txt = generate_txt_for_registration(txt_path, txt_format, task_output_path)
    return setting_folder_path, output_pair_list_txt


def generate_txt_for_registration(txt_path, txt_format,output_path):
    if txt_format =="aug_by_file":
        output_pair_list_txt = get_pair_list_txt_by_file(txt_path,output_path)
    else:
        output_pair_list_txt = get_pair_list_txt_by_line(txt_path,output_path)
    return output_pair_list_txt



def do_registration(txt_path, setting_folder_path,output_path,gpu_id):
    cmd = "python demo_for_easyreg_eval "
    cmd +="-ts={} -txt={} -o={} -g={}".format(setting_folder_path,txt_path,output_path,gpu_id)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()


def do_data_augmentation(args):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param registration_pair_list:  list of registration pairs, [source_list, target_list, lsource_list, ltarget_list]
    :return: None
    """
    task_output_path = args.task_output_path
    gpu_id = args.gpu_id
    setting_folder_path, output_pair_list_txt=init_aug_env(args)
    do_registration(output_pair_list_txt,setting_folder_path,task_output_path,gpu_id)
    shooting_new_data(task_output_path,gpu_id,)


if __name__ == '__main__':
    """
    A data augmentation interface for optimization methods or learning methods with pre-trained models.
    
    In the case that a lot of unlabeled data is available, we suggest to train a network via demo_for_easyreg_train.py,
    which would provide fast interpolation for data augmentation. Otherwise, the optimization option is recommended.
    
    As the registration precision is not highly necessary for data-augmentation, the default setting is fine.
    Of course, feel free to fine tune the multi_gaussian_stds if the task is to perform a precise pair interpolation
    
    Though the purpose of this script is to provide demo, it is a generalized interface for fluid-based data augmentation
    The method support list: mermaid-related ( optimizing/pretrained) methods
    
    IMPORTANT !!!!!!!!!!!!!!!!!
    Currently, we assume the all the input pairs are affinely aligned.
    (Todo to remove this constrain, the geodesic space should either be 1D or an atlas image needs to be introduced for two-step registration)
    
    Two input formats are supported:
    1) aug_by_line: input txt where each line refer to a source image, a target set and the source label (option), the labels of target sets(option)
    the augmentation takes place for each line
    2) aug_by_file: input txt where each line refer to a image and corresponding label (option)
    the augmentation takes place among lines
    
    All the settings should be given in the setting folder.
    Though we support both learning-based and optimization based registration, for the learning-based method,  
    a mermaid-setting file with pretrained model path should be provided, please refer to the demo we provide here for details.

    Arguments:
        demo related:
             --run_demo: run the demo
             --demo_name: opt_lddmm/learnt_lddmm
             --txt_path/-txt: the input txt file
        other arguments:
             --setting_folder_path/-ts :path of the folder where settings are saved
             --task_output_path/ -o: the path of output folder
             --gpu_id/ -g: gpu_id to use


    """
    import argparse

    parser = argparse.ArgumentParser(description='An easy interface for evaluate various registration methods')
    parser.add_argument("--run_demo", required=False, action='store_true', help='run demo')
    parser.add_argument('--demo_name', required=False, type=str, default='opt_lddmm',
                        help='opt_lddmm/learnt_lddmm')
    # ---------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-ts', '--setting_folder_path', required=False, type=str,
                        default=None,
                        help='path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings(optional) and mermaid_nonp_settings(optional)')
    parser.add_argument('-txt_path', '--t', required=False, default=None, type=str,
                        help='the txt file recording the pairs to registration')  # 2
    parser.add_argument('-txt_format', '--f', required=False, default="aug_by_file", type=str,
                        help='txt format, aug_by_line/aug_by_file')
    parser.add_argument('-o', "--task_output_path", required=True, default=None, help='the output path')
    parser.add_argument('-g', "--gpu_id", required=False, type=int, default=0, help='gpu_id to use')

    args = parser.parse_args()
    print(args)
    run_demo = args.run_demo
    demo_name = args.demo_name
    txt_path = args.txt_path
    txt_format = args.txt_format

    if run_demo:
        assert demo_name in ["opt_lddmm","learnt_lddmm"]
    assert os.path.isfile(txt_path),"file not exist"
    assert txt_format in ["aug_by_line","aug_by_file"]
    do_data_augmentation(args)
