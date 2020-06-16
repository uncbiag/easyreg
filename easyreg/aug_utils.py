import random
from easyreg.reg_data_utils import write_list_into_txt
import os
from easyreg.reg_data_utils import read_fname_list_from_pair_fname_txt, generate_pair_name

def gen_inter_pair_list(img_path_list,fname_list,label_path_list, pair_num_limit=-1, per_num_limit=-1):
    """
    :param img_path_list: Nx1 list
    :param label_path_list: Nx1 list
    :param pair_num_limit: max number of the random pair
    :param per_num_limit: max number of a image being the source image
    :return:
    """
    img_pair_list = []
    pair_name_list = []
    num_img = len(img_path_list)
    for i in range(num_img):
        img_pair_list_tmp = []
        pair_name_list_tmp = []
        for j in range(num_img):
            if i != j:
                if label_path_list[i][j] is not None:
                    img_pair_list_tmp.append([img_path_list[i], img_path_list[j],
                                              label_path_list[i], label_path_list[j]])
                else:
                    img_pair_list_tmp.append([img_path_list[i], img_path_list[j]])
                if fname_list is not None:
                    pair_name_list_tmp.append([fname_list[i] + "_" + fname_list[j],fname_list[i],fname_list[j]])
                else:
                    pair_name_list_tmp.append(generate_pair_name([img_path_list[i], img_path_list[j]],detail=True))
        if len(img_pair_list_tmp) > per_num_limit and per_num_limit>-1:
            ind = list(range(len(img_pair_list_tmp)))
            random.shuffle(ind)
            img_pair_list_tmp = [img_pair_list_tmp[ind[i]] for i in range(per_num_limit)]
            pair_name_list_tmp  = [pair_name_list_tmp[ind[i]] for i in range(per_num_limit)]
        img_pair_list += img_pair_list_tmp
        pair_name_list += pair_name_list_tmp
    if len(img_pair_list)>pair_num_limit and pair_num_limit >= 0:
        ind = list(range(len(img_pair_list)))
        random.shuffle(ind)
        img_pair_list = [img_pair_list[ind[i]] for i in range(pair_num_limit)]
        pair_name_list = [pair_name_list[ind[i]] for i in range(pair_num_limit)]
        return img_pair_list, pair_name_list
    else:
        return img_pair_list, pair_name_list


def gen_intra_pair_list(img_path_list,fname_list,label_path_list, pair_num_limit=-1, per_num_limit=-1):
    """
    :param img_path_list: NxK list
    :param label_path_list: NxK list
    :param pair_num_limit: max number of the random pair
    :param per_num_limit: max number of a image being the source image
    :return:
    """
    img_pair_list = []
    pair_name_list = []
    num_img = len(img_path_list)
    for i in range(num_img):
        img_pair_list_tmp = []
        pair_name_list_tmp = []
        for j in range(1,len(img_path_list[i])):
            if label_path_list[i][j] is not None:
                img_pair_list_tmp.append([img_path_list[i][0], img_path_list[i][j],
                                          label_path_list[i][0], label_path_list[i][j]])
            else:
                img_pair_list_tmp.append([img_path_list[i][0], img_path_list[i][j]])
            if fname_list is not None:
                pair_name_list_tmp.append([fname_list[i][0] + "_" + fname_list[i][j],fname_list[i][0],fname_list[i][j]])
            else:
                pair_name_list_tmp.append(generate_pair_name([img_path_list[i][0], img_path_list[i][j]],detail=True))
        if len(img_pair_list_tmp) > per_num_limit and per_num_limit > -1:
            ind = list(range(len(img_pair_list_tmp)))
            random.shuffle(ind)
            img_pair_list_tmp = [img_pair_list_tmp[ind[i]] for i in range(per_num_limit)]
            pair_name_list_tmp = [pair_name_list_tmp[ind[i]] for i in range(per_num_limit)]
        img_pair_list += img_pair_list_tmp
        pair_name_list += pair_name_list_tmp
    if len(img_pair_list) > pair_num_limit and pair_num_limit >= 0:
        ind = list(range(len(img_pair_list)))
        random.shuffle(ind)
        img_pair_list = [img_pair_list[ind[i]] for i in range(pair_num_limit)]
        pair_name_list = [pair_name_list[ind[i]] for i in range(pair_num_limit)]
        return img_pair_list, pair_name_list
    else:
        return img_pair_list, pair_name_list




def gen_post_aug_pair_list(test_img_path_list,train_img_path_list, test_fname_list=None,train_fname_list=None,
                           test_label_path_list=None,train_label_path_list=None, pair_num_limit=-1, per_num_limit=-1):
    """

    :param test_img_path_list:
    :param train_img_path_list:
    :param test_fname_list:
    :param train_fname_list:
    :param test_label_path_list:
    :param train_label_path_list:
    :param pair_num_limit:
    :param per_num_limit:
    :return:
    """
    img_pair_list = []
    pair_name_list = []
    num_test_img = len(test_img_path_list)
    num_train_img = len(train_img_path_list)
    if test_label_path_list is None:
        test_label_path_list = ["None"]*num_test_img
    if train_label_path_list is None:
        train_label_path_list = ["None"]*num_train_img

    for i in range(num_test_img):
        img_pair_list_tmp = []
        pair_name_list_tmp = []
        for j in range(num_train_img):
            img_pair_list_tmp.append([test_img_path_list[i], train_img_path_list[j],
                                      test_label_path_list[i], train_label_path_list[j]])
            if train_fname_list is not None and test_fname_list is not None:
                pair_name_list_tmp.append([test_fname_list[i] + "_" + train_fname_list[j],test_fname_list[i], train_fname_list[j]])
            else:
                pair_name_list_tmp.append(generate_pair_name([test_img_path_list[i], train_img_path_list[j]],detail=True))
        if len(img_pair_list_tmp) > per_num_limit and per_num_limit>-1:
            ind = list(range(len(img_pair_list_tmp)))
            random.shuffle(ind)
            img_pair_list_tmp = [img_pair_list_tmp[ind[i]] for i in range(per_num_limit)]
            pair_name_list_tmp  = [pair_name_list_tmp[ind[i]] for i in range(per_num_limit)]
        img_pair_list += img_pair_list_tmp
        pair_name_list += pair_name_list_tmp
    if len(img_pair_list)>pair_num_limit and pair_num_limit >= 0:
        ind = list(range(len(img_pair_list)))
        random.shuffle(ind)
        img_pair_list = [img_pair_list[ind[i]] for i in range(pair_num_limit)]
        pair_name_list = [pair_name_list[ind[i]] for i in range(pair_num_limit)]
        return img_pair_list, pair_name_list
    else:
        return img_pair_list, pair_name_list


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





def get_pair_list_txt_by_file(file_txt,name_txt,output_path,pair_num_limit=-1, per_num_limit=-1):
    pair_list_path = os.path.join(output_path, "pair_path_list.txt")
    pair_name_path = os.path.join(output_path, "pair_name_list.txt")
    if not os.path.isfile(pair_list_path):
        img_label_list = read_img_label_into_list(file_txt)
        num_image = len(img_label_list)
        img_list = [img_label_list[i][0] for i in range(num_image)]
        label_list = [img_label_list[i][1] for i in range(num_image)]
        if name_txt is not None:
            fname_list = read_fname_list_from_pair_fname_txt(name_txt)
        else:
            fname_list = None
        pair_list,pair_name_list = gen_inter_pair_list(img_list,fname_list,label_list,pair_num_limit,per_num_limit)
        write_list_into_txt(pair_list_path, pair_list)
        write_list_into_txt(pair_name_path, pair_name_list)
    else:
        print("the file {} has already exist, now read it".format(pair_list_path))
    return pair_list_path,pair_name_path


def get_pair_list_txt_by_line(file_txt,name_txt,output_path,pair_num_limit=-1, per_num_limit=-1):
    pair_list_path = os.path.join(output_path,"pair_path_list.txt")
    pair_name_path = os.path.join(output_path,"pair_name_list.txt")
    if not os.path.isfile(pair_list_path):
        img_label_list = read_img_label_into_list(file_txt)
        num_img = len(img_label_list)
        set_size_list = [int(len(img_label)/2) for img_label in img_label_list]
        img_list = [[img_label_list[i][j] for j in range(set_size_list[i])] for i in range(num_img)]
        label_list = [[img_label_list[i][j+set_size_list[i]] for j in range(set_size_list[i])] for i in range(num_img)]
        if name_txt is not None:
            fname_list = read_fname_list_from_pair_fname_txt(name_txt,detail=True)
        else:
            fname_list = None
        pair_list,pair_name_list = gen_intra_pair_list(img_list,fname_list,label_list,pair_num_limit,per_num_limit)
        write_list_into_txt(pair_list_path,pair_list)
        write_list_into_txt(pair_name_path,pair_name_list)
    else:
        print("the file {} has already exist, now read it".format(pair_list_path))
    return pair_list_path,pair_name_path




def generate_moving_target_dict(pair_path_list_txt,pair_name_list_txt=None):
    pair_path_list = read_img_label_into_list(pair_path_list_txt)
    if pair_name_list_txt is None:
        pair_name_detail_list = [generate_pair_name([pair_path[0],pair_path[1]],detail=True) for pair_path in pair_path_list]
    else:
        pair_name_detail_list = read_fname_list_from_pair_fname_txt(pair_name_list_txt,detail=True)
    moving_name_list = [name_list[1] for name_list in pair_name_detail_list]
    moving_name_set = set(moving_name_list)
    moving_target_dict = {moving_name:{'m_pth':None,"t_pth":[],"name":[],"l_pth":None} for moving_name in moving_name_set}
    for i in range(len(moving_name_list)):
        has_label = len(pair_path_list[i]) == 4
        moving_name = moving_name_list[i]
        moving_target_dict[moving_name]['m_pth']=pair_path_list[i][0]
        moving_target_dict[moving_name]["t_pth"].append(pair_path_list[i][1])
        moving_target_dict[moving_name]["name"].append(pair_name_detail_list[i])
        if has_label:
            moving_target_dict[moving_name_list[i]]["l_pth"] = pair_path_list[i][2]

    return moving_target_dict

def generate_moving_momentum_txt(pair_path_list_txt,momentum_path,output_path_txt_path,output_name_txt_path,pair_name_list_txt,affine_path):
    moving_target_dict = generate_moving_target_dict(pair_path_list_txt,pair_name_list_txt)
    moving_momentum_list =[]
    fname_list = []
    for moving_name, item in moving_target_dict.items():
        label_path = "None" if item['l_pth'] is None else item['l_pth']
        fname_list.append([item["name"][0][1]] + [name[2] for name in item["name"]])
        momentum_name_list = [name[0] +"_0000_Momentum.nii.gz" for name in item["name"]]
        momentum_path_list = [os.path.join(momentum_path, momentum_name) for momentum_name in momentum_name_list]
        affine_path_list = []
        if affine_path is not None:
            affine_path_list = [os.path.join(affine_path,name[0]+"_affine_param.npy") for name in item["name"]]
        moving_momentum_list_tmp = [item['m_pth']] + [label_path] + momentum_path_list + affine_path_list
        moving_momentum_list.append(moving_momentum_list_tmp)
    write_list_into_txt(output_path_txt_path,moving_momentum_list)
    write_list_into_txt(output_name_txt_path,fname_list)


def split_txt(input_txt,num_split, output_folder,sub_fname="p"):
    import math
    os.makedirs(output_folder,exist_ok=True)
    pairs = read_img_label_into_list(input_txt)
    if len(pairs):
        if isinstance(pairs[0],list):
            pairs = [["None" if item is None else item for item in pair] for pair in pairs]
        else:
            pairs = ["None" if item is None else item for item in pairs]
    num_pairs = len(pairs)
    chunk_size = max(math.ceil(num_pairs/num_split),1)
    index = list(range(0, num_pairs, chunk_size))
    for i, ind in enumerate(index):
        split = pairs[ind:ind+chunk_size]
        write_list_into_txt(os.path.join(output_folder, '{}{}.txt'.format(sub_fname,i)),split)
    return len(index)

