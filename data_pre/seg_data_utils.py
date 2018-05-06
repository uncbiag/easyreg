from os import listdir
from os.path import isfile, join
from glob import glob
import os
from scipy import misc
import numpy as np
import h5py
import skimage
import SimpleITK as sitk
import sys
import random

PYTHON_VERSION = 3
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2
import data_pre.fileio as fileio
import data_pre.module_parameters as pars


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
            f_filter = glob(f_path, recursive=True)
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
    if len(label_path):
        return [os.path.join(label_path, fn_switch(os.path.split(file_path)[1])) for file_path in file_path_list]
    else:
        return [fn_switch(file_path) for file_path in file_path_list]


def make_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist



def get_file_name(file_path,last_ocur=False):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    return name



def divide_data_set(root_path, file_path_list, ratio, m_mod=None):
    """
    divide the dataset into root_path/train root_path/val root_path/test
    :param root_path: the root path for saving the task_dataset
    :param file_name_list: list of name of the saved file  like img1
    :param ratio: tuple of (train_ratio, val_ratio, test_ratio) from all the files
    :return:  full path of each file

    """
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    file_name_list = [get_file_name(fp) for fp in file_path_list]
    file_num = len(file_name_list)
    sub_path = {x: os.path.join(root_path, x) for x in ['train', 'val', 'test','debug']}
    nt = [make_dir(sub_path[key]) for key in sub_path]
    if sum(nt):
        raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n"
                         "manually delete the folder to reprepare the data")
    train_num = int(train_ratio * file_num)
    val_num = int(val_ratio * file_num)
    file_name_sub_list = {}
    fp_sub_list ={}
    file_name_sub_list['train'] = file_name_list[:train_num]
    file_name_sub_list['val'] = file_name_list[train_num: train_num + val_num]
    file_name_sub_list['test'] = file_name_list[train_num + val_num:]
    file_name_sub_list['debug'] = file_name_list[: val_num]
    fp_sub_list['train'] = file_path_list[:train_num]
    fp_sub_list['val'] = file_path_list[train_num: train_num + val_num]
    fp_sub_list['test'] = file_path_list[train_num + val_num:]
    fp_sub_list['debug'] = file_path_list[: val_num]
    if m_mod is None:
        saving_path_list = [os.path.join(sub_path[x], file_name) for x in ['train', 'val', 'test','debug'] for file_name
                        in file_name_sub_list[x]]
    else:
        saving_path_list = [os.path.join(os.path.join(sub_path[x], m_mod),file_name) for x in ['train', 'val', 'test', 'debug'] for
                            file_name in file_name_sub_list[x]]

    saving_path_dic= {fn: saving_path_list[i] for i, fn in enumerate (file_name_list)}
    saving_path_debug_dic = {fn+'_debug': os.path.join(sub_path['debug'], fn) for fn in file_name_sub_list['debug']}
    saving_path_dic.update(saving_path_debug_dic)
    return saving_path_dic, fp_sub_list


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
    acutally design for read oai txt
    :param file_path:
    :return:
    """
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [line.split(',') for line in content]
        paths=[item[0] for item in lists]
        file_name_list = [os.path.split(path)[1] for path in paths]
        lists= [item.split('.')[0] for item in file_name_list]
    return lists


def str_concat(lists,linker='_'):
    from functools import reduce
    str_concated = reduce((lambda x,y:str(x)+linker+str(y)), lists)
    return str_concated



def extract_train_patch_info(patch_dic):
    label = str(patch_dic['label']) if 'label' in patch_dic else str(-1)
    start_coord = str_concat(patch_dic['start_coord']) if 'start_coord' in patch_dic else str_concat((-1,-1,-1))
    label_threshold = "{:.5f}".format(patch_dic['threshold']) if 'threshold' in patch_dic else str(-1)
    return {'l':label, 'sc':start_coord,'th':label_threshold}

def extract_test_patch_info(patches_dic):
    tile_size = patches_dic['tile_size']
    tile_size = str_concat(tile_size)
    overlap_size = patches_dic['overlap_size']
    overlap_size = str_concat(overlap_size)
    padding_mode = patches_dic['padding_mode']

    return {'tile_size':tile_size, 'overlap_size':overlap_size,'padding_mode':padding_mode}


def gen_fn_from_info(info):
    info_list = ['_'+key+'_'+str(value) for key,value in info.items()]
    file_name = str_concat(info_list) if len(info_list) else ''
    return file_name



def get_patch_saving_path(file_path,info, saving_by_patch=True):
    img_name = get_file_name(file_path)
    if saving_by_patch:
        file_name = gen_fn_from_info(info)
        file_name = img_name + file_name
        label_str = info['l']
        folder_path=os.path.join(file_path,label_str)
        make_dir(folder_path)
        full_path = os.path.join(folder_path,file_name+'.h5py')
        file_id = file_name
    else:
        file_name = img_name
        make_dir(file_path)
        full_path = os.path.join(file_path,file_name+'.h5py')
        file_id = file_name

    return full_path, file_id

def saving_patch_per_img(patch,file_path,itk_img=True):

    info = extract_train_patch_info(patch)
    patch_img = sitk_to_np(patch['img']) if itk_img else patch['img']
    if 'seg' in patch:
        patch_seg = sitk_to_np(patch['seg']) if itk_img else patch['seg']
    else:
        patch_seg = np.array([-1])
    saving_path, patch_id = get_patch_saving_path(file_path,info)
    save_to_h5py(saving_path,patch_img.astype(np.float32), patch_seg.astype(np.int32),patch_id,info,verbose=False)

def saving_patches_per_img(patches,file_path):

    info = extract_test_patch_info(patches)
    patches_img = patches['img']
    if 'seg' in patches:
        patches_seg = patches['seg']
    else:
        patches_seg= np.array([-1])
    saving_path, file_id = get_patch_saving_path(file_path,info, saving_by_patch=False)
    save_to_h5py(saving_path,patches_img.astype(np.float32), patches_seg.astype(np.int32) ,file_id,info,verbose=False)

def saving_per_img(sample,file_path,info={}):
    # info = info
    # img = sitk_to_np(sample['img'])
    # if 'seg' in sample:
    #     seg = sitk_to_np(sample['seg'])
    # else:
    #     seg = np.array([-1])
    saving_path, file_id = get_patch_saving_path(file_path,info, saving_by_patch=False)

    if not isinstance(sample['img'], list):
        sitk.WriteImage(sitk.Cast(sample['img'], sitk.sitkFloat32), saving_path.replace('.h5py', '_tmod_'+str(0)+'.nii.gz'))
    else:
        [sitk.WriteImage(sitk.Cast(im, sitk.sitkFloat32), saving_path.replace('.h5py', '_tmod_'+str(i)+'.nii.gz')) for i, im in enumerate(sample['img'])]
    if 'seg' in sample:
        sitk.WriteImage(sitk.Cast(sample['seg'], sitk.sitkInt32), saving_path.replace('.h5py', '_seg.nii.gz'))
    save_to_h5py(saving_path,np.array([-1]), np.array([-1]),file_id,info,verbose=False)




def check_same_size(img, standard):
    """
    make sure all the image are of the same size
    :param img: img to compare
    :param standard: standarded image size
    """
    assert img.shape == standard, "img size must be the same"


def np_to_sitk(img, info=None):
    """

    :param img: a list or a single numpy imge
    :param info:
    :return: if the input image is a list, then return a list of sitk img other wise return single sitk image
    """
    if not isinstance(img,list):
        sitk_img = sitk.GetImageFromArray(img)
    else:
        sitk_img =[sitk.GetImageFromArray(im) for im in img]
    if info is not None:
        pass   # will implemented later
    return sitk_img


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



def normalize_img(image, sched='tp'):
    """
    normalize image,
    warning, default [-1,1], which would be tricky when dealing with the bilinear,
    which default background is 0
    :param sched: 'ntp': percentile 0.95 then normalized to [-1,1] 'tp': percentile then [0,1], 'p' percentile
\   :return: normalized image
    """
    if sched == 'ntp':
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    elif sched == 'tp':
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif sched == 'p':
        image[:] = image / np.percentile(image, 95) * 0.95
    elif sched == 't':
        image[:] = (image - np.min(image)) / (np.max(image) - np.min(image))


def file_io_read_img(path, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True,
                     adaptive_padding=16):
    ##########################################################################################
    normalize_intensities = False if is_label else normalize_intensities
    ##########################################################################################
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image,
                                                                 adaptive_padding, inverse=True)
    if normalize_spacing:
        spacing = normalized_spacing
    else:
        spacing = spacing

    info = {'spacing': spacing, 'img_size': im.shape}
    return im, info


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
    par['info'][('num_label', info['num_label'], 'num of label')]
    par['info'][('label_density', info['label_density'], 'label_density')]
    par['info'][('standard_label_index', info['standard_label_index'], 'standard_label_index')]
    par['info'][('sample_data_path',info['sample_data_path'],'sample_data_path')]
    par['info'][('sample_label_path',info['sample_label_path'],'sample_label_path')]
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
        if dic['data'] is not None:
            f.create_dataset('data', data=dic['data'].astype(np.float32))
        if dic['label'] is not None:
            f.create_dataset('label', data=dic['label'].astype(np.int32))
        for key, value in dic['info'].items():
            f.attrs[key] = value
        # asciiList = [[path.encode("ascii", "ignore") for path in file] for file in dic['file_path']]
        asciiList = dic['file_id'].encode("ascii", "ignore")
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('file_id', data=asciiList, dtype=string_dt)
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
    dic = {'data': img, 'info': info, 'file_id': file_id, 'label': label}
    write_file(path, dic, type='h5py')
    if verbose:
        print('data saved: {}'.format(path))
        print(dic['info'])
        print("the shape of file{}".format(dic['data'][:].shape))
