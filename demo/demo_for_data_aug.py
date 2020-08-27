"""
A demo for fluid-based data augmentation.
"""

import matplotlib as matplt
import subprocess
matplt.use('Agg')
import os, sys, time
import torch
torch.backends.cudnn.benchmark=True

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../easy_reg'))
# sys.path.insert(0,os.path.abspath('../mermaid'))
import tools.module_parameters as pars
from easyreg.aug_utils import *




def generate_txt_for_registration(file_txt,name_txt, txt_format,output_path,pair_num_limit=-1,per_num_limit=-1):
    if txt_format =="aug_by_file":
        pair_list_txt, pair_name_list = get_pair_list_txt_by_file(file_txt,name_txt,output_path,pair_num_limit,per_num_limit)
    else:
        pair_list_txt, pair_name_list = get_pair_list_txt_by_line(file_txt,name_txt,output_path,pair_num_limit,per_num_limit)
    return pair_list_txt, pair_name_list

def init_reg_env(args):
    task_output_path = args.task_output_path
    run_demo = args.run_demo
    name_txt = args.name_txt

    if run_demo:
        demo_name = args.demo_name
        setting_folder_path = os.path.join('./demo_settings/data_aug', demo_name)
        task_output_path = os.path.join('./data_aug_demo_output', demo_name)
        args.task_output_path = task_output_path
        file_txt = os.path.join(task_output_path,"input.txt")
        txt_format = "aug_by_line" if demo_name=="opt_lddmm_lpba" else "aug_by_file"
    else:
        txt_format = args.txt_format
        file_txt = args.file_txt
        setting_folder_path = args.setting_folder_path
    os.makedirs(task_output_path, exist_ok=True)
    output_pair_list_txt, output_name_list_txt = generate_txt_for_registration(file_txt,name_txt, txt_format, task_output_path,args.max_size_of_pair_to_reg,args.max_size_of_target_set_to_reg)
    return setting_folder_path, output_pair_list_txt, output_name_list_txt




def do_registration(txt_path, name_path, setting_folder_path,output_path,gpu_id_list):
    if len(gpu_id_list)==1:
        cmd = "python demo_for_easyreg_eval.py "
        cmd +="-ts={} -txt={} -pntxt={} -o={} -g={}".format(setting_folder_path,txt_path,name_path,output_path,int(gpu_id_list[0]))
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
    else:
        num_split = len(gpu_id_list)
        num_split = split_txt(txt_path, num_split, output_path, "p")
        split_txt(name_path, num_split, output_path, "pn")
        sub_txt_path_list = [os.path.join(output_path, 'p{}.txt'.format(i)) for i in range(num_split)]
        sub_name_path_list = [os.path.join(output_path, 'pn{}.txt'.format(i)) for i in range(num_split)]
        processes = []
        for i in range(num_split):
            cmd = "echo GPU {} \n".format(gpu_id_list[i])
            cmd += "python demo_for_easyreg_eval.py "
            cmd += "-ts={} -txt={} -pntxt={} -o={} -g={}\n".format(setting_folder_path, sub_txt_path_list[i],sub_name_path_list[i], output_path, int(gpu_id_list[i]))
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
            time.sleep(60)
        exit_codes = [p.wait() for p in processes]



def init_aug_env(reg_pair_list_txt,reg_name_list_txt,task_output_path,setting_folder_path):
    aug_output_path = os.path.join(task_output_path, "aug")
    os.makedirs(aug_output_path,exist_ok=True)
    aug_setting_path = os.path.join(setting_folder_path, "data_aug_setting.json")
    aug_setting = pars.ParameterDict()
    aug_setting.load_JSON(aug_setting_path)
    fluid_mode = aug_setting["data_aug"]["fluid_aug"]["fluid_mode"]
    reg_res_folder_path = os.path.join(task_output_path,"reg/res/records")
    aug_input_txt = os.path.join(aug_output_path,"aug_input_path.txt")
    aug_name_txt = os.path.join(aug_output_path,"aug_input_name.txt")
    affine_path = None
    if fluid_mode == "aug_with_nonaffined_data":
        affine_path = reg_res_folder_path
    if fluid_mode == "aug_with_atlas":
        aug_setting["data_aug"]["fluid_aug"]["to_atlas_folder"] = reg_res_folder_path
        aug_setting["data_aug"]["fluid_aug"]["atlas_to_folder"] = reg_res_folder_path
        aug_setting.write_JSON(aug_setting_path)
    generate_moving_momentum_txt(reg_pair_list_txt,reg_res_folder_path,aug_input_txt,aug_name_txt,reg_name_list_txt,affine_path)
    return aug_input_txt,aug_name_txt,aug_output_path



def do_augmentation(input_txt, input_name_txt, setting_folder_path, aug_output_path,gpu_list):
    aug_setting_path = os.path.join(setting_folder_path,"data_aug_setting.json")
    mermaid_setting_path = os.path.join(setting_folder_path,"mermaid_nonp_settings.json")
    assert os.path.isfile(aug_setting_path), "the aug setting json  {} is not found".format(aug_setting_path)
    aug_setting = pars.ParameterDict()
    aug_setting.load_JSON(aug_setting_path)
    task_type = aug_setting["data_aug"]["fluid_aug"]["task_type"]
    num_process = len(gpu_list)
    if task_type == "rand_sampl":
        max_aug_num = aug_setting["data_aug"]["max_aug_num"]
        max_aug_num_per_process = round(max_aug_num/num_process)
        aug_setting["data_aug"]["max_aug_num"]=max_aug_num_per_process
        aug_setting_mp_path = os.path.join(setting_folder_path,"data_aug_setting_mutli_process.json")
        aug_setting.write_ext_JSON(aug_setting_mp_path)
        processes = []
        for i in range(num_process):
            cmd = "python gen_aug_samples.py "
            cmd += "-t={} -n={} -as={} -ms={} -o={} -g={}\n".format(input_txt,input_name_txt,aug_setting_mp_path,mermaid_setting_path,aug_output_path,int(gpu_list[i]))
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
            time.sleep(1)


    else:
        num_process = split_txt(input_txt, num_process, aug_output_path,"aug_p")
        split_txt(input_name_txt, num_process, aug_output_path,"aug_np")
        sub_input_txt_list = [os.path.join(aug_output_path, 'aug_p{}.txt'.format(i)) for i in range(num_process)]
        sub_input_name_txt_list = [os.path.join(aug_output_path, 'aug_np{}.txt'.format(i)) for i in range(num_process)]
        processes = []
        for i in range(num_process):
            cmd = "python gen_aug_samples.py "
            cmd += "-t={} -n={} -as={} -ms={} -o={} -g={}\n".format(sub_input_txt_list[i],sub_input_name_txt_list[i], aug_setting_path, mermaid_setting_path,
                                                           aug_output_path,int(gpu_list[i]))
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
            time.sleep(1)


    exit_codes = [p.wait() for p in processes]







def pipeline(args):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param registration_pair_list:  list of registration pairs, [source_list, target_list, lsource_list, ltarget_list]
    :return: None
    """
    setting_folder_path, reg_pair_list_txt, reg_name_list_txt=init_reg_env(args)
    do_registration(reg_pair_list_txt,reg_name_list_txt, setting_folder_path,args.task_output_path,args.gpu_id_list)
    aug_input_txt,aug_name_txt, aug_output_path = init_aug_env(reg_pair_list_txt,reg_name_list_txt,args.task_output_path,setting_folder_path)
    do_augmentation(aug_input_txt,aug_name_txt,setting_folder_path, aug_output_path, args.gpu_id_list)


if __name__ == '__main__':
    """
    Though the purpose of this script is to provide demo, it is a generalized interface for fluid-based data augmentation and interpolation

    
    The augmentation/interpolatin include two parts 
    1. do fluid-based registration with either optimization methods or learning methods with pre-trained models.
    2. do data augmentation via random sampling on geodesic space and time axis
       or do data inter-/extra-polation with given direction and time point from geodesic space
    
    it will call two script:  demo_for_easyreg_eval.py and gen_aug_samples.py
    so setting files for both tasks are need to be provided, see demo for details
    
    As high precision registration is not necessary for data-augmentation, the default setting will be fine for most cases.
    Of course, feel free to fine tune the multi_gaussian_stds and iterations if the task is to do data interpolation.
    
    In the case that a lot of unlabeled data is available, we suggest to train a network via demo_for_easyreg_train.py first,
    which would provide fast interpolation for data augmentation. Otherwise, the optimization option is recommended.
        
    
    For file_txt, two input formats are supported:
    1) aug_by_line: input txt where each line refer to a path of source image, paths of target images and the source label (string "None" if not exist), the labels of target images(None if not exist)
    the augmentation takes place for each line
    2) aug_by_file: input txt where each line refer to a image and corresponding label (string "None" if not exist)
    the augmentation takes place among lines
    
    For the name_txt (optional, will use the filename if not provided) include the fname for each image( to avoid confusion of source images with the same filename)
    1) aug_by_line: each line include a source name,  target names
    2) aug_by_file: each line include a image name
    
    
    All the settings should be given in the setting folder.
    We support both learning-based and optimization based registration, 
    for the learning-based method, the pretrained model path should be provided in cur_task_setting.json
    Arguments:
        demo related:
             --run_demo: run the demo
             --demo_name: opt_lddmm_lpba/learnt_lddmm_oai
             --gpu_id_list/ -g: gpu_id_list to use
        other arguments:
             --file_txt/-txt: the input txt recording the file path
             --name_txt/-txt: the input txt recording the file name
             --txt_format: aug_by_file/aug_by_line
             --max_size_of_target_set_to_reg: max size of the target set for each source image, set -1 if there is no constraint
             --max_size_of_pair_to_reg: max size of pair for registration, set -1 if there is no constraint, in that case the potential pair  number would be N*(N-1) if txt_format is set as aug_by_file
             --setting_folder_path/-ts :path of the folder where settings are saved
             --task_output_path/ -o: the path of output folder


    """
    import argparse
    parser = argparse.ArgumentParser(description='An easy interface for evaluate various registration methods')
    parser.add_argument("--run_demo", required=False, action='store_true', help='run demo')
    parser.add_argument('--demo_name', required=False, type=str, default='opt_lddmm_lpba',
                        help='opt_lddmm_lpba/learnt_lddmm_oai')
    # ---------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-ts', '--setting_folder_path', required=False, type=str,
                        default="",
                        help='path of the folder where settings are saved,should include cur_task_setting.json,data_aug_setting, mermaid_affine_settings(optional) and mermaid_nonp_settings(optional)')
    parser.add_argument('-t','--file_txt',  required=False, default="", type=str,
                        help='the txt file recording the file to augment')
    parser.add_argument('-n', '--name_txt', required=False, default=None, type=str,
                        help='the txt file recording the corresponding file name')
    parser.add_argument('-f','--txt_format',  required=False, default="aug_by_file", type=str,
                        help='txt format, aug_by_line/aug_by_file')
    parser.add_argument('-mt','--max_size_of_target_set_to_reg',  required=False, default=10, type=int,
                        help='max size of the target set for each source image, set -1 if there is no constraint')
    parser.add_argument('-ma','--max_size_of_pair_to_reg', required=False, default=-1, type=int,
                        help='max size of pair for registration, set -1 if there is no constraint, in that case the potential pair  number would be N*(N-1) if txt_format is set as aug_by_file')
    parser.add_argument('-o', "--task_output_path", required=False, default="",  type=str,help='the output path')
    parser.add_argument('-g', "--gpu_id_list", nargs='+', required=False, default=None, help='list of gpu id to use')

    args = parser.parse_args()
    print(args)
    run_demo = args.run_demo
    demo_name = args.demo_name
    file_txt = args.file_txt
    txt_format = args.txt_format

    if run_demo:
        assert demo_name in ["opt_lddmm_lpba","learnt_lddmm_oai","learnt_lddmm_oai_interpolation"]
    assert os.path.isfile(file_txt) or run_demo,"file not exist"
    assert txt_format in ["aug_by_line","aug_by_file"]
    pipeline(args)
