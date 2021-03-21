import os
from easyreg.reg_data_utils import loading_img_list_from_files,generate_pair_name

local_path = "/playpen-raid/zyshen/oai_data"
server_path = "/pine/scr/z/y/zyshen/data/oai_data"
switcher = (local_path, server_path)


def server_switcher(f_path,switcher=("","")):
    if len(switcher[0]):
        f_path = f_path.replace(switcher[0],switcher[1])
    return f_path

def split_input(original_txt_path):
    source_path_list, target_path_list, l_source_path_list, l_target_path_list = loading_img_list_from_files(
        original_txt_path)
    file_num = len(source_path_list)
    if l_source_path_list is not None and l_target_path_list is not None:
        assert len(source_path_list) == len(l_source_path_list)
        file_list = [[source_path_list[i], target_path_list[i],l_source_path_list[i],l_target_path_list[i]] for i in range(file_num)]
    else:
        file_list = [[source_path_list[i], target_path_list[i]] for i in range(file_num)]
    fname_list = [generate_pair_name([file_list[i][0],file_list[i][1]]) for i in range(file_num)]
    return file_list, fname_list

def print_txt(txt, output_path):
    with open(output_path, "w") as f:
        f.write(txt)

def printer(cmdl, name,output_path, mem=6, n_cpu=8, t=3):

    txt = """#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem={}g
#SBATCH -n 1
#SBATCH -c {}
#SBATCH --output={}.txt
#SBATCH -t {}:00:00


source activate torch4
cd /pine/scr/z/y/zyshen/reg_clean/demo
srun {}
""".format(mem,n_cpu,name,t,cmdl)
    print_txt(txt,output_path)
    return output_path

def generate_slurm_txt(task_path_list,txt_output_folder):
    txt = "#!/bin/bash\n"
    slurm_txt_path = os.path.join(txt_output_folder,"slurm.sh")
    txt_output_folder_sever = server_switcher(txt_output_folder,switcher)
    for task_path in task_path_list:
        txt += "sbatch {}/{}\n".format(txt_output_folder_sever,task_path)
    print_txt(txt, slurm_txt_path)


def get_cmdl(setting_folder_path,task_output_path,pair_path,gpu_id=0):
    cmdl = "python demo_for_easyreg_eval.py --setting_folder_path {} --task_output_path {} -s {}  -t {} -ls {} -lt {} --gpu_id {}"\
        .format(setting_folder_path,task_output_path,pair_path[0],pair_path[1], pair_path[2], pair_path[3],gpu_id)
    return cmdl


def generate_pair_setting(setting_folder_path,task_output_folder,pair_path_list,pair_name_list,txt_output_folder,switcher=("",""), mem=12, n_cpu=8, t=36, gpu_id=0):
    server_task_list = []
    for pair_path, pair_name in zip(pair_path_list, pair_name_list):
        pair_path = [server_switcher(path,switcher) for path in pair_path]
        task_pair_output_path = os.path.join(task_output_folder,pair_name)
        cmdl = get_cmdl(setting_folder_path, task_pair_output_path, pair_path, gpu_id)
        pair_txt_name = pair_name+".sh"
        txt_output_path = os.path.join(txt_output_folder,pair_txt_name)
        printer(cmdl, pair_name, txt_output_path, mem, n_cpu, t)
        server_task_list.append(pair_txt_name)
    return server_task_list




def generate_inter_setting(task_type):

    task_name = "{}_inter".format(task_type)
    setting_folder_path = os.path.join(server_path,"task_settings_for_full_resolution",task_type)
    pair_txt_path = os.path.join(local_path,"reg_debug_labeled_oai_reg_inter","test","pair_path_list.txt")
    txt_output_folder = os.path.join(local_path,"sever_slurm",task_name)
    task_output_folder = os.path.join(server_path,"expri",task_name)
    os.makedirs(txt_output_folder,exist_ok=True)
    file_list, fname_list = split_input(pair_txt_path)
    server_task_list = generate_pair_setting(setting_folder_path,task_output_folder,file_list,fname_list,txt_output_folder,switcher=switcher, gpu_id=-1)
    generate_slurm_txt(server_task_list,txt_output_folder)

def generate_atlas_setting(task_type):
    local_path = "/playpen-raid/zyshen/oai_data"
    server_path = "/pine/scr/z/y/zyshen/data/oai_data"
    task_name = "{}_atlas".format(task_type)
    setting_folder_path = os.path.join(server_path,"task_settings_for_full_resolution",task_type)
    pair_txt_path = os.path.join(local_path,"reg_test_for_atlas","test","pair_path_list.txt")
    txt_output_folder = os.path.join(local_path,"sever_slurm",task_name)
    task_output_folder = os.path.join(server_path,"expri",task_name)
    os.makedirs(txt_output_folder,exist_ok=True)
    file_list, fname_list = split_input(pair_txt_path)
    server_task_list = generate_pair_setting(setting_folder_path,task_output_folder,file_list,fname_list,txt_output_folder,switcher=switcher, gpu_id=-1)
    generate_slurm_txt(server_task_list,txt_output_folder)



task_types = ["ants","demons","nifty_reg"]

for task_type in task_types:
    generate_inter_setting(task_type)
    generate_atlas_setting(task_type)


