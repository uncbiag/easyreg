import random
from easyreg.reg_data_utils import write_list_into_txt
import os

def gen_inter_pair_list(img_path_list,label_path_list, pair_num_limit=-1, per_num_limit=-1):
    """
    :param img_path_list: Nx1 list
    :param label_path_list: Nx1 list
    :param pair_num_limit: max number of the random pair
    :param per_num_limit: max number of a image being the source image
    :return:
    """
    img_pair_list = []
    num_img = len(img_path_list)
    for i in range(num_img):
        img_pair_list_tmp = []
        for j in range(num_img):
            if i != j:
                if label_path_list[i][j] is not None:
                    img_pair_list_tmp.append([img_path_list[i], img_path_list[j],
                                              label_path_list[i], label_path_list[j]])
                else:
                    img_pair_list_tmp.append([img_path_list[i], img_path_list[j]])
        if len(img_pair_list_tmp) > per_num_limit and per_num_limit>-1:
            img_pair_list_tmp = random.sample(img_pair_list_tmp, per_num_limit)
        img_pair_list += img_pair_list_tmp
    if pair_num_limit >= 0:
        random.shuffle(img_pair_list)
        return img_pair_list[:pair_num_limit]
    else:
        return img_pair_list


def gen_intra_pair_list(img_path_list,label_path_list, pair_num_limit=-1, per_num_limit=-1):
    """
    :param img_path_list: NxK list
    :param label_path_list: NxK list
    :param pair_num_limit: max number of the random pair
    :param per_num_limit: max number of a image being the source image
    :return:
    """
    img_pair_list = []
    num_img = len(img_path_list)
    for i in range(num_img):
        img_pair_list_tmp = []
        for j in range(1,len(img_path_list[i])):
            if label_path_list[i][j] is not None:
                img_pair_list_tmp.append([img_path_list[i][0], img_path_list[i][j],
                                          label_path_list[i][0], label_path_list[i][j]])
            else:
                img_pair_list_tmp.append([img_path_list[i][0], img_path_list[i][j]])
        if len(img_pair_list_tmp) > per_num_limit and per_num_limit>-1:
            img_pair_list_tmp = random.sample(img_pair_list_tmp, per_num_limit)
        img_pair_list += img_pair_list_tmp
    if pair_num_limit >= 0:
        random.shuffle(img_pair_list)
        return img_pair_list[:pair_num_limit]
    else:
        return img_pair_list


def read_img_label_into_list(file_path):
    """
    read the list from the file, each elem in a line compose a list, each line compose to a list,
    the elem "None" would be filtered and not considered
    :param file_path: the file path to read
    :return: list of list
    """
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            #lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists


def get_pair_list_txt_by_file(txt_path,output_path,pair_num_limit=-1, per_num_limit=-1):
    output_txt_path = os.path.join(output_path,"pair_to_reg.txt")
    if not os.path.isfile(output_txt_path):
        img_label_list = read_img_label_into_list(txt_path)
        num_image = len(img_label_list)
        img_list = [img_label_list[i][0] for i in range(num_image)]
        label_list = [img_label_list[i][1] for i in range(num_image)]
        img_pair_list = gen_inter_pair_list(img_list,label_list,pair_num_limit,per_num_limit)
        write_list_into_txt(output_txt_path,img_pair_list)
    else:
        print("the file {} has already exist, now read it".format(output_txt_path))
    return output_txt_path


def get_pair_list_txt_by_line(txt_path,output_path,pair_num_limit=-1, per_num_limit=-1):
    output_txt_path = os.path.join(output_path,"pair_to_reg.txt")
    if not os.path.isfile(output_txt_path):
        img_label_list = read_img_label_into_list(txt_path)
        num_img = len(img_label_list)
        set_size_list = [int(len(img_label)/2) for img_label in img_label_list]
        img_list = [[img_label_list[i][j] for j in range(set_size_list[i])] for i in range(num_img)]
        label_list = [[img_label_list[i][j+set_size_list[i]] for j in range(set_size_list[i])] for i in range(num_img)]
        img_pair_list = gen_intra_pair_list(img_list,label_list,pair_num_limit,per_num_limit)
        write_list_into_txt(output_txt_path,img_pair_list)
    else:
        print("the file {} has already exist, now read it".format(output_txt_path))
    return output_txt_path



from easyreg.reg_data_utils import *

def generate_moving_target_dict(txt_path):
    pair_list = read_img_label_into_list(txt_path)
    moving_name_list = [get_file_name(pth[0]) for pth in pair_list]
    moving_name_set = set(moving_name_list)
    moving_target_dict = {moving_name:{'m_pth':None,"t_pth":[],"l_pth":None} for moving_name in moving_name_set}
    for i in range(len(moving_name_list)):
        has_label = len(pair_list[i]) == 4
        moving_target_dict[moving_name_list[i]]['m_pth']=pair_list[i][0]
        moving_target_dict[moving_name_list[i]]["t_pth"].append(pair_list[i][1])
        if has_label:
            moving_target_dict[moving_name_list[i]]["l_pth"] = pair_list[i][2]

    return moving_target_dict

def generate_moving_momentum_txt(txt_path, momentum_path, output_txt_path,affine_path=None):
    moving_target_dict = generate_moving_target_dict(txt_path)
    moving_momentum_list =[]
    for moving_name, item in moving_target_dict.items():
        label_path = "None" if item['l_pth'] is None else item['l_pth']
        #momentum_name_list = [moving+'_'+get_file_name(t) +"_0000_Momentum.nii.gz" for t in item['t_pth']]
        momentum_name_list = [moving_name+'_'+get_file_name(t) +"_0000_Momentum.nii.gz" for t in item['t_pth']]
        momentum_path_list = [os.path.join(momentum_path, momentum_name) for momentum_name in momentum_name_list]
        affine_path_list = []
        if affine_path is not None:
            affine_path_list = [os.path.join(affine_path,moving_name+'_'+get_file_name(t)+"_affine_param.npy") for t in item['t_pth']]
        moving_momentum_list_tmp = [item['m_pth']] + [label_path] + momentum_path_list + affine_path_list
        moving_momentum_list.append(moving_momentum_list_tmp)
    write_list_into_txt(output_txt_path,moving_momentum_list)


def split_txt(input_txt,num_split, output_folder,sub_fname="p"):
    import math
    os.makedirs(output_folder,exist_ok=True)
    pairs = read_img_label_into_list(input_txt)
    pairs = [["None" if item is None else item for item in pair] for pair in pairs]
    num_pairs = len(pairs)
    chunk_size = max(math.ceil(num_pairs/num_split),1)
    index = list(range(0, num_pairs, chunk_size))
    for i, ind in enumerate(index):
        split = pairs[ind:ind+chunk_size]
        write_list_into_txt(os.path.join(output_folder, '{}{}.txt'.format(sub_fname,i)),split)
    return len(index)

