from os import listdir
from os.path import isfile, join
from glob import glob
import os
import numpy as np
import h5py
import sys
PYTHON_VERSION = 3
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2
import data_pre.fileio as fileio
import tools.module_parameters as pars




def list_dic(path):
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]



def list_pairwise(path, img_type, full_comb, sched):
    """
     return the list of  paths of the paired image  [N,2]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
    :param sched: sched can be inter personal or intra personal
    :return:
    """
    if sched == 'intra':
        dic_list = list_dic(path)
        pair_list = intra_pair(path, dic_list, img_type, full_comb)
    elif sched == 'inter':
        pair_list = inter_pair(path, img_type, full_comb)
    else:
        raise ValueError("schedule should be 'inter' or 'intra'")
    return pair_list

def check_full_comb_on(full_comb):
    if full_comb == False:
        print("only return the pair in order, to get more pairs, set the 'full_comb' True")
    else:
        print(" 'full_comb' is on, if you don't need all possible pair, set the 'full_com' False")


def inter_pair(path, type, full_comb=False, mirrored=False):
    """
    get the paired filename list
    :param path: dic path
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
    :param mirrored: double the data,  generate pair2_pair1 from pair1_pair2
    :return: [N,2]
    """
    check_full_comb_on(full_comb)
    pair_list=[]
    for sub_type in type:
        f_path = join(path,'**', sub_type)
        if PYTHON_VERSION == 3: #python3
            f_filter = glob(f_path, recursive=True)
        else:
            f_filter = []
            import fnmatch
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, sub_type):
                    f_filter.append(os.path.join(root, filename))

        f_num = len(f_filter)
        if not full_comb:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:  # too many pairs , easy to out of memory
            raise ValueError("Warnning, too many pairs, be sure the disk is big enough. Comment this line if you want to continue ")
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    if mirrored:
        pair_list = mirror_pair(pair_list)
    return pair_list



def mirror_pair(pair_list):
    """
    double the data,  generate pair2_pair1 from pair1_pair2    :param pair_list:
    :return:
    """
    return pair_list + [[pair[1],pair[0]] for pair in pair_list]



def intra_pair(path, dic_list, type, full_comb, mirrored=False):
    """

    :param path: dic path
    :param dic_list: each elem in list contain the path of folder which contains the instance from the same person
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param full_comb:  if full_comb, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
    check_full_comb_on(full_comb)
    pair_list = []
    for dic in dic_list:
        if PYTHON_VERSION == 3:
            f_path = join(path, dic, type[0])
            f_filter = glob(f_path)
        else:
            f_filter = []
            f_path = join(path, dic)
            import fnmatch
            for root, dirnames, filenames in os.walk(f_path):
                for filename in fnmatch.filter(filenames, type[0]):
                    f_filter.append(os.path.join(root, filename))
        f_num = len(f_filter)
        assert f_num != 0
        if not full_comb:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    if mirrored:
        pair_list = mirror_pair(pair_list)
    return pair_list


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
            f_filter = glob(f_path, recursive=True)
        else:
            f_filter = []
            import fnmatch
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, sub_type):
                    f_filter.append(os.path.join(root, filename))
    return f_filter


def find_corr_pair_map(pair_path_list, label_path):
    """
    get the label path from the image path, assume the file name is the same
    :param pair_path_list: the path list of the image
    :param label_path: the path of the label folder
    :return:
    """
    return [[os.path.join(label_path, os.path.split(pth)[1]) for pth in pair_path] for pair_path in pair_path_list]

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


def str_concat(lists,linker='_'):
    from functools import reduce
    str_concated = reduce((lambda x,y:str(x)+linker+str(y)), lists)
    return str_concated

# def divide_data_set(root_path, pair_name_list, ratio):
#     """
#     divide the dataset into root_path/train root_path/val root_path/test
#     :param root_path: the root path for saving the task_dataset
#     :param pair_name_list: list of name of the saved pair  like img1_img2
#     :param ratio: tuple of (train_ratio, val_ratio, test_ratio) from all the pairs
#     :return:  full path of each file
#
#     """
#     train_ratio = ratio[0]
#     val_ratio = ratio[1]
#     pair_num = len(pair_name_list)
#     sub_path = {x:os.path.join(root_path,x) for x in ['train', 'val', 'test']}
#     nt = [make_dir(sub_path[key]) for key in sub_path]
#     if sum(nt):
#         raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n" \
#                          "manually delete the folder to reprepare the data")
#     train_num = int(train_ratio * pair_num)
#     val_num = int(val_ratio*pair_num)
#     pair_name_sub_list={}
#     pair_name_sub_list['train'] = pair_name_list[:train_num]
#     pair_name_sub_list['val'] = pair_name_list[train_num: train_num+val_num]
#     pair_name_sub_list['test'] = pair_name_list[train_num+val_num:]
#     saving_path_list = [os.path.join(sub_path[x],pair_name+'.h5py') for x in ['train', 'val', 'test'] for pair_name in pair_name_sub_list[x] ]
#     return saving_path_list

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
    divided_path_dic['pair_path_list'] ={sess:[pair_path_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    divided_path_dic['pair_name_list'] ={sess:[pair_name_list[idx] for idx in file_id_dic[sess]] for sess in sesses}
    return divided_path_dic






def saving_pair_info(sub_folder_dic, divided_path_dic):
    """
    saved the image pair into txt
    :param sub_folder_dic: dict of path , i.e. {'train':train_saving_folder_pth, 'val': val_saving_folder_pth, ....}
    :param divided_path_dic:  dict of image pairs i.e. {
                                                 'pair_path_list':{'train':[[s1, t1], [s2, t2]....], 'val': [[s1, t1], [s2, t2]....]}....},
                                                 'pair_name_list':{'train':[s1_t1, s2_ts,....], 'val':[s1_t1, s2_t2].....}
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

def read_txt_into_list(file_path):
    """
    read the list from the file, each elem in a line compose a list, each line compose to a list,
    the elem "None" would be filtered and not considered
    :param file_path: the file path to read
    :return: list of list
    """
    assert os.path.isfile(file_path), "the file {} doesnt exist".format(file_path)
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists

def read_fname_list_from_pair_fname_txt(file_path,detail=False):
    """
    the txt file may has two type
    1) 1 item per line,  that is the pair_name
    2) 2 item per line, that is the pair_name  moving_name  target_name
    :param file_path:
    :param detail:
    :return:
    """
    fname_list = read_txt_into_list(file_path)
    if len(fname_list) and isinstance(fname_list[0],list):
        if detail:
            return fname_list
        else:
            return [fname[0] for fname in fname_list]
    else:
        return fname_list


def loading_img_list_from_files(path):
    """
    loading image pair list from the file
    :param path: path of the file to read
    :return: source_path_list, target_path_list, l_source_path_list ( None if there is no label), l_target_path_list (None if there is no label)
    """
    path_list = read_txt_into_list(path)
    num_pair = len(path_list)
    assert len(path_list[0])>=2
    source_path_list = [path_list[i][0] for i in range(num_pair)]
    target_path_list = [path_list[i][1] for i in range(num_pair)]
    l_source_path_list = [path_list[i][2] if len((path_list[i]))==4 else "None" for i in range(num_pair)]
    l_target_path_list = [path_list[i][3] if len((path_list[i]))==4 else "None" for i in range(num_pair)]
    return source_path_list, target_path_list, l_source_path_list, l_target_path_list


def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name

def generate_pair_name(pair_path,detail=False):
    source_path, target_path = pair_path
    f = lambda x: os.path.split(x)
    #assert source_path != target_path,"the source image should be different to the target image"
    while True:
        s = get_file_name(f(source_path)[-1])
        t = get_file_name(f(target_path)[-1])
        if s !=t:
            break
        else:
            source_path, target_path = f(source_path)[0],f(target_path)[0]
    pair_name = s+"_"+t
    if not detail:
        return pair_name
    else:
        return pair_name, s, t

def check_same_size(img, standard):
    """
    make sure all the image are of the same size
    :param img: img to compare
    :param standard: standarded image size
    """
    assert img.shape == standard, "img size must be the same"





def file_io_read_img(path, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True, adaptive_padding=4):
    normalize_intensities = False if is_label else normalize_intensities
    ################################################################################################################################################
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, False, squeeze_image,adaptive_padding)
    ################################################################################################################################################
    if normalize_spacing:
        spacing = normalized_spacing
    else:
        spacing = spacing
    info = { 'spacing':spacing, 'img_size': im.shape}
    return im, info

def file_io_read_img_slice(path, slicing, axis, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True,adaptive_padding=4):
    """

    :param path: file path
    :param slicing: int, the nth slice of the img would be sliced
    :param axis: int, the nth axis of the img would be sliced
    :param is_label:  the img is label
    :param normalize_spacing: normalized the spacing
    :param normalize_intensities: normalized the img
    :param squeeze_image:
    :param adaptive_padding: padding the img to favored size, (divided by certain number, here is 4), here using default 4 , favored by cuda fft
    :return:
    """
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image,adaptive_padding)
    if normalize_spacing:
        spacing = normalized_spacing
    else:
        spacing = spacing

    if axis == 1:
        slice = im[slicing]
        slicing_spacing = spacing[1:]
    elif axis == 2:
        slice = im[:,slicing,:]
        slicing_spacing = np.asarray([spacing[0], spacing[2]])
    elif axis == 3:
        slice = im[:,:,slicing]
        slicing_spacing = spacing[:2]
    else:
        raise ValueError("slicing axis exceed, should be 1-3")
    info = { 'spacing':slicing_spacing, 'img_size': slice.shape}
    return slice, info



def save_sz_sp_to_json(info, output_path):
    """
    save img size and img spacing info into json
    :param info:
    :param output_path:
    :return:
    """
    par = pars.ParameterDict()
    par[('info', {}, 'shared information of data')]
    par['info'][('img_sz', info['img_size'], 'size of image')]
    par['info'][('spacing', info['spacing'].tolist(), 'size of image')]
    if 'num_label' in info:
        par['info'][('num_label', info['num_label'], 'num of label')]
    if 'standard_label_index' in info:
        par['info'][('standard_label_index', info['standard_label_index'], 'standard_label_index')]
    par.write_JSON(os.path.join(output_path, 'info.json'))




def read_h5py_file(path, type='h5py'):
    """
    return dictionary contain 'data' and   'info': start_coord, end_coord, file_path
    :param path:
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'r')
        data = f['data'][:]
        info = {}
        label = None
        if '/label' in f:
            label = f['label'][:]
        for key in f.attrs:
            info[key] = f.attrs[key]
        #info['file_id'] = f['file_id'][:]
        f.close()
        return {'data': data, 'info': info, 'label': label}


def write_file(path, dic, type='h5py'):
    """

    :param path: file path
    :param dic:  which has three item : numpy 'data', numpy 'label'if exists,  dic 'info' , string list 'file_path',
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.create_dataset('data', data=dic['data'])
        if dic['label'] is not None:
            f.create_dataset('label', data=dic['label'])
        for key, value in dic['info'].items():
            f.attrs[key] = value
        # asciiList = [[path.encode("ascii", "ignore") for path in file] for file in dic['file_path']]
        # asciiList = dic['file_id'].encode("ascii", "ignore")
        # string_dt = h5py.special_dtype(vlen=str)
        # f.create_dataset('file_id', data=asciiList, dtype=string_dt)
        f.close()
    else:
        raise ValueError('only h5py supported currently')




def save_to_h5py(path, img,label,file_id, info, verbose=True):
    """

    :param path:  path for saving file
    :param img_file_list: list of image file
    :param info: additional info
    :param img_file_path_list:  list of path/name of image file
    :param img_file_path_list:  list of path/name of corresponded label file

    :return:
    """
    info['file_id'] =file_id
    dic = {'data': img, 'info': info, 'label': label}
    write_file(path, dic, type='h5py')
    if verbose:
        print('data saved: {}'.format(path))
        print(dic['info'])
        print("the shape of file{}".format(dic['data'][:].shape))



