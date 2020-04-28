from os import listdir
from os.path import isfile, join
from glob import glob
import os
import numpy as np
import h5py
import SimpleITK as sitk
import sys

PYTHON_VERSION = 3
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2


def list_dic(path):
    return [dic for dic in listdir(path) if not isfile(join(path, dic))]


def get_file_path_list(path, img_type):
    """
     return the list of  paths of the image  [N,1]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param full_comb: if full_comb, return all possible files, if not, return files in increasing order
    :param sched: sched can be inter personal or intra personal
    :return:
    """
    f_filter=[]
    for sub_type in img_type:
        if PYTHON_VERSION == 3:  # python3
            f_path = join(path, '**', sub_type)
            f_filter += glob(f_path, recursive=True)
        else:
            f_filter = []
            import fnmatch
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, sub_type):
                    f_filter.append(os.path.join(root, filename))
    return f_filter




def get_multi_mode_path(path, multi_mode_list):
    return  [path.replace(multi_mode_list[0],mode) for mode in multi_mode_list]


def find_corr_map(file_path_list, label_path, label_switch = ('','')):
    """
    get the label path from the image path
    :param file_path_list: the path list of the image
    :param label_path: the path of the label folder
    :return:
    """
    fn_switch = lambda x: x.replace(label_switch[0], label_switch[1])

    if label_path is not None:
        label_path_list =  [os.path.join(label_path, fn_switch(os.path.split(file_path)[1])) for file_path in file_path_list]
    else:
        label_path_list = [ fn_switch(file_path) for file_path in file_path_list]
    return label_path_list


def make_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist



def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name

def divide_data_set(root_path, pair_num,ratio):
    """
    divide the dataset into root_path/train root_path/val root_path/test root_path/debug
    :param root_path: the root path for saving the task_dataset
    :param pair_num: num of pair
    :param ratio: tuple of (train_ratio, val_ratio, test_ratio) from all the pairs
    :return:  full path of each file

    """
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    sub_folder_dic = {x:os.path.join(root_path,x) for x in ['train', 'val', 'test','debug']}
    # debug details maybe added later
    make_dir(os.path.join(root_path,'debug'))
    nt = [make_dir(sub_folder_dic[key]) for key in sub_folder_dic]

    # if sum(nt):
    #     raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n" \
    #                      "manually delete the folder to reprepare the data")
    train_num = int(train_ratio * pair_num)
    val_num = int(val_ratio*pair_num)
    file_id_dic={}
    file_id_dic['train'] = list(range(train_num))
    file_id_dic['val'] = list(range(train_num, train_num+val_num))
    file_id_dic['test'] = list(range(train_num+val_num,pair_num))
    file_id_dic['debug'] = list(range(train_num))
    return sub_folder_dic, file_id_dic





def get_divided_dic(file_id_dic, pair_path_list, pair_name_list):
    """
    get the set dict of the image pair path and pair name

    :param file_id_dic: dict of set index, {'train':[1,2,3,..100], 'val':[101,...120]......}
    :param pair_path_list: list of pair_path, [[s1,t1], [s2,t2],.....]
    :param pair_name_list: list of fnmae [s1_t1, s2_t2,....]
    :return: divided_path_dic {'pair_path_list':{'train': [[s1,t1],[s2,t2]..],'val':[..],...}, 'pair_name_list':{'train':[s1_t1, s2_t2,...],'val':[..],..}}
    """
    divided_path_dic = {}
    sesses = ['train','val','test','debug']
    divided_path_dic['file_path_list'] ={sess:[pair_path_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    divided_path_dic['file_name_list'] ={sess:[pair_name_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    return divided_path_dic



def load_file_path_from_txt(root_path,file_path_list, txt_path):
    """
     return the list of  paths of the image  [N,1]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param full_comb: if full_comb, return all possible files, if not, return files in increasing order
    :param sched: sched can be inter personal or intra personal
    :return:
    """
    sub_path = {x: os.path.join(root_path, x) for x in ['train', 'val', 'test', 'debug']}
    nt = [make_dir(sub_path[key]) for key in sub_path]
    # if sum(nt):
    #     raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n"
    #                      "manually delete the folder to reprepare the data")
    file_name_sub_list = {}
    sesses = ['train','val','test','debug']
    fp_sub_list = {}
    file_path_dic = {get_file_name(fp):fp for fp in file_path_list}
    for sess in sesses:
        file_name_sub_list[sess]= read_txt_into_list(os.path.join(txt_path, sess+'.txt'))
        fp_sub_list[sess] = [file_path_dic[fp] for fp in file_name_sub_list[sess]]

    saving_path_dic={file_name:os.path.join(sub_path[x], file_name) for x in ['train', 'val', 'test'] for file_name
                        in file_name_sub_list[x]}
    saving_path_debug_dic = {fn+'_debug': os.path.join(sub_path['debug'], fn) for fn in file_name_sub_list['debug']}
    saving_path_dic.update(saving_path_debug_dic)
    return saving_path_dic, fp_sub_list



def read_txt_into_list(file_path):
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
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists



def saving_pair_info(sub_folder_dic, divided_path_dic):
    """
    saved the image pair into txt
    :param sub_folder_dic: dict of path , i.e. {'train':train_saving_folder_pth, 'val': val_saving_folder_pth, ....}
    :param divided_path_dic:  dict of image pairs i.e. {
                                                 'pair_path_list':{'train':[s1, s2....], 'val': [s1, s2....]}....},
                                                 'pair_name_list':{'train':[f1, ...], 'val':[f1,...].....}
                                                 }
    :return: None
    """
    for sess, sub_folder_path in sub_folder_dic.items():
        for item, list_to_write in divided_path_dic.items():
            file_path = os.path.join(sub_folder_path,item+'.txt')
            write_list_into_txt(file_path, list_to_write[sess])






def write_list_into_txt(file_path, list_to_write):
    """
    write the list into txt,  each elem refers to a line
    if elem is also a list, then each sub elem is separated by the space

    :param file_path: the file path to write in
    :param list_to_write: the list to refer
    :return: None
    """
    with open(file_path, 'w') as f:
        if len(list_to_write):
            if isinstance(list_to_write[0],(float, int, str)):
                f.write("\n".join(list_to_write))
            elif isinstance(list_to_write[0],(list, tuple)):
                new_list = ["     ".join(sub_list) for sub_list in list_to_write]
                f.write("\n".join(new_list))
            else:
                raise(ValueError,"not implemented yet")

def sitk_to_np(img, info=None):
    """

    :param img:   img is either a list  or a single itk image
    :param info:
    :return: if the input image is a list, then return a list of numpy image otherwise return single numpy image
    """

    if not isinstance(img, list):
        np_img = [sitk.GetArrayViewFromImage(img)]
    else:
        np_img = [sitk.GetArrayViewFromImage(im) for im in img]
    np_img = np.stack(np_img,0)
    if info is not None:
        pass   # will implemented later
    return np_img

