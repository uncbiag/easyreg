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
from easyreg.aug_utils import *



def init_reg_env(args):
    task_output_path = args.task_output_path
    run_demo = args.run_demo
    if run_demo:
        demo_name = args.demo_name
        setting_folder_path = os.path.join('./demo_settings/data_aug', demo_name)
        task_output_path = os.path.join('./demo_data_aug', demo_name)
        args.task_output_path = task_output_path
        txt_path = os.path.join(task_output_path,"source_target_set.txt")
        txt_format = "aug_by_line"
    else:
        txt_format = args.txt_format
        txt_path = args.txt_path
        setting_folder_path = args.setting_folder_path
    os.makedirs(task_output_path, exist_ok=True)
    output_pair_list_txt = generate_txt_for_registration(txt_path, txt_format, task_output_path,args.max_size_of_pair_to_reg,args.max_size_of_target_set_to_reg)
    return setting_folder_path, output_pair_list_txt


def init_aug_env(reg_pair_list_txt,output_path,setting_folder_path):
    aug_setting_path = os.path.join(setting_folder_path, "data_aug_setting.json")
    aug_setting = pars.ParameterDict()
    aug_setting.load_JSON(aug_setting_path)
    fluid_mode = aug_setting["data_aug"]["fluid_aug"]["fluid_mode"]
    reg_res_folder_path = os.path.join(output_path,"reg/res/records")
    aug_input_txt = os.path.join(output_path,"moving_momentum.txt")
    affine_path = None
    if fluid_mode == "aug_with_nonaffined_data":
        affine_path = reg_res_folder_path
    if fluid_mode == "aug_with_atlas":
        aug_setting["data_aug"]["fluid_aug"]["to_atlas_folder"] = reg_res_folder_path
        aug_setting["data_aug"]["fluid_aug"]["atlas_to_folder"] = reg_res_folder_path
        aug_setting.write_JSON(aug_setting_path)
    generate_moving_momentum_txt(reg_pair_list_txt,reg_res_folder_path,aug_input_txt,affine_path)
    return aug_input_txt


def generate_txt_for_registration(txt_path, txt_format,output_path,pair_num_limit=-1,per_num_limit=-1):
    if txt_format =="aug_by_file":
        output_pair_list_txt = get_pair_list_txt_by_file(txt_path,output_path,pair_num_limit,per_num_limit)
    else:
        output_pair_list_txt = get_pair_list_txt_by_line(txt_path,output_path,pair_num_limit,per_num_limit)
    return output_pair_list_txt

def do_augmentation(input_txt, setting_folder_path, task_output_path):
    aug_setting_path = os.path.join(setting_folder_path,"data_aug_setting.json")
    mermaid_setting_path = os.path.join(setting_folder_path,"mermaid_nonp_settings.json")
    assert os.path.isfile(aug_setting_path), "the aug setting json  {} is not found".format(aug_setting_path)
    aug_setting = pars.ParameterDict()
    aug_setting.load_JSON(aug_setting_path)
    task_type = aug_setting["data_aug"]["fluid_aug"]["task_type"]
    num_process = 5
    if task_type == "rand_aug":
        max_aug_num = aug_setting["data_aug"]["max_aug_num"]
        max_aug_num_per_process = round(max_aug_num/num_process)
        aug_setting["data_aug"]["data_aug"]=max_aug_num_per_process
        aug_setting_mp_path = os.path.join(setting_folder_path,"data_aug_setting_mutli_process.json")
        aug_setting.write_ext_JSON(aug_setting_mp_path)
        cmd = ""
        for _ in range(num_process):
            cmd += "python gen_aug_samples.py "
            cmd += "-txt={}  -as={} -ms={} -o={} & \n".format(input_txt,aug_setting_mp_path,mermaid_setting_path,task_output_path)
            cmd += "sleep 1s \n"

    else:
        num_process = split_txt(txt_path, num_process, task_output_path,"aug_p")
        sub_input_txt_list = [os.path.join(task_output_path, 'aug_p{}.txt'.format(i)) for i in range(num_process)]
        cmd = ""
        for i in range(num_process):
            cmd += "python gen_aug_samples.py "
            cmd += "-txt={}  -as={} -ms={} -o={} & \n".format(sub_input_txt_list[i], aug_setting_path, mermaid_setting_path,
                                                           task_output_path)
            cmd += "sleep 1s \n"

    process = subprocess.Popen(cmd, shell=True)
    process.wait()






def do_registration(txt_path, setting_folder_path,output_path,gpu_id_list):
    cmd = ""
    if len(gpu_id_list)==1:
        cmd += "python demo_for_easyreg_eval.py "
        cmd +="-ts={} -txt={} -o={} -g={}".format(setting_folder_path,txt_path,output_path,int(gpu_id_list[0]))
    else:
        num_split = len(gpu_id_list)
        split_txt(txt_path, num_split, output_path)
        sub_txt_path_list = [os.path.join(output_path, 'p{}.txt'.format(i)) for i in range(num_split)]
        for i,sub_txt_path in enumerate(sub_txt_path_list):
            cmd += "echo GPU {} \n".format(gpu_id_list[i])
            cmd += "python demo_for_easyreg_eval.py "
            cmd += "-ts={} -txt={} -o={} -g={} & \n".format(setting_folder_path, sub_txt_path, output_path, int(gpu_id_list[i]))
            cmd += "sleep 1m \n"

    process = subprocess.Popen(cmd, shell=True)
    process.wait()


def pipeline(args):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param registration_pair_list:  list of registration pairs, [source_list, target_list, lsource_list, ltarget_list]
    :return: None
    """
    setting_folder_path, reg_pair_list_txt=init_reg_env(args)
    do_registration(reg_pair_list_txt,setting_folder_path,args.task_output_path,args.gpu_id_list)
    aug_input_txt = init_aug_env(reg_pair_list_txt,args.task_output_path,setting_folder_path)
    do_augmentation(aug_input_txt,setting_folder_path, args.task_output_path)


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
    1) aug_by_line: input txt where each line refer to a source image, a target set and the source label (None if not exist), the labels of target sets(None if not exist)
    the augmentation takes place for each line
    2) aug_by_file: input txt where each line refer to a image and corresponding label (None if not exist)
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
             --gpu_id_list/ -g: gpu_id_list to use


    """
    import argparse
    # --run_demo --demo_name=opt_lddmm_lpba
    # -ts=/playpen-raid1/zyshen/debug/xu/opt_lddmm -t=/playpen-raid1/zyshen/debug/xu/source_target_set.txt -f=aug_by_line -o=/playpen-raid1/zyshen/debug/xu/expr -g 0 0 1 1

    parser = argparse.ArgumentParser(description='An easy interface for evaluate various registration methods')
    parser.add_argument("--run_demo", required=False, action='store_true', help='run demo')
    parser.add_argument('--demo_name', required=False, type=str, default='opt_lddmm_lpba',
                        help='opt_lddmm_lpba/learnt_lddmm_oai')
    # ---------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-ts', '--setting_folder_path', required=False, type=str,
                        default="",
                        help='path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings(optional) and mermaid_nonp_settings(optional)')
    parser.add_argument('-t','--txt_path',  required=False, default="", type=str,
                        help='the txt file recording the pairs to registration')  # 2
    parser.add_argument('-f','--txt_format',  required=False, default="aug_by_file", type=str,
                        help='txt format, aug_by_line/aug_by_file')
    parser.add_argument('-mt','--max_size_of_target_set_to_reg',  required=False, default=5, type=int,
                        help='max size of the target set for each source image, set -1 if there is no constraint')
    parser.add_argument('-ma','--max_size_of_pair_to_reg', required=False, default=-1, type=int,
                        help='max size of pair for registration, set -1 if there is no constraint, in that case the potential pair  number would be N*(N-1) if txt_format is set as aug_by_file')
    parser.add_argument('-o', "--task_output_path", required=False, default="",  type=str,help='the output path')
    parser.add_argument('-g', "--gpu_id_list", nargs='+', required=False, default=None, help='list of gpu id to use')

    args = parser.parse_args()
    print(args)
    run_demo = args.run_demo
    demo_name = args.demo_name
    txt_path = args.txt_path
    txt_format = args.txt_format

    if run_demo:
        assert demo_name in ["opt_lddmm_lpba","learnt_lddmm_oai"]
    assert os.path.isfile(txt_path) or run_demo,"file not exist"
    assert txt_format in ["aug_by_line","aug_by_file"]
    pipeline(args)
