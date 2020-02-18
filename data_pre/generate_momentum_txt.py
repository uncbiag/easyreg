import os
from easyreg.reg_data_utils import *

def generate_moving_target_dict(txt_path):
    pair_list = read_txt_into_list(txt_path)
    moving_name_list = [get_file_name(pth[0]) for pth in pair_list]
    has_label = len(pair_list[0]) == 4
    moving_name_set = set(moving_name_list)
    moving_target_dict = {moving_name:{'m_pth':None,"t_pth":[],"l_pth":None} for moving_name in moving_name_set}
    for i in range(len(moving_name_list)):
        moving_target_dict[moving_name_list[i]]['m_pth']=pair_list[i][0]
        moving_target_dict[moving_name_list[i]]["t_pth"].append(pair_list[i][1])
        if has_label:
            moving_target_dict[moving_name_list[i]]["l_pth"] = pair_list[i][2]

    return moving_target_dict, has_label

def generate_moving_momentum_txt(txt_path, momentum_path, output_txt_path):
    moving_target_dict, has_label = generate_moving_target_dict(txt_path)
    moving_momentum_list =[]
    for moving, item in moving_target_dict.items():
        label_path = "None" if not has_label else item['l_pth']
        #momentum_name_list = [moving+'_'+get_file_name(t) +"_0000_Momentum.nii.gz" for t in item['t_pth']]
        momentum_name_list = [moving+'_'+get_file_name(t) +"_0000Momentum.nii.gz" for t in item['t_pth']]
        momentum_path_list = [os.path.join(momentum_path, momentum_name) for momentum_name in momentum_name_list]
        moving_momentum_list_tmp = [item['m_pth']] + [label_path] + momentum_path_list
        moving_momentum_list.append(moving_momentum_list_tmp)
    write_list_into_txt(output_txt_path,moving_momentum_list)




num_c_list=[10,30]
for num_c in num_c_list:
    txt_path="/playpen-raid/zyshen/data/oai_reg/train_with_{}/train/pair_path_list.txt".format(num_c)
    momentum_path = "/playpen-raid/zyshen/data/oai_reg/train_with_{}/momentum_lresol".format(num_c)
    output_txt_path = "/playpen-raid/zyshen/data/oai_reg/train_with_{}/momentum_lresol.txt".format(num_c)
    generate_moving_momentum_txt(txt_path, momentum_path, output_txt_path)
#
# num_c_list=[5,10,15,20,25]
# for num_c in num_c_list:
#     txt_path="/playpen-raid/zyshen/data/lpba_reg/train_with_{}/train/pair_path_list.txt".format(num_c)
#     momentum_path = "/playpen-raid/zyshen/data/lpba_reg/train_with_25/lpba_ncc_reg1/momentum_lresol".format(num_c)
#     output_txt_path = "/playpen-raid/zyshen/data/lpba_reg/train_with_{}/lpba_ncc_reg1/momentum_lresol.txt".format(num_c)
#     generate_moving_momentum_txt(txt_path,momentum_path,output_txt_path)
#

