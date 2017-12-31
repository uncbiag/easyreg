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


def get_filename_list(path, img_type):
    """
     return the list of  paths of the image  [N,1]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param full_comb: if full_comb, return all possible files, if not, return files in increasing order
    :param sched: sched can be inter personal or intra personal
    :return:
    """
    for sub_type in img_type:
        f_path = join(path, '**', sub_type)
        if PYTHON_VERSION == 3:  # python3
            f_filter = glob(f_path, recursive=True)
        else:
            f_path_list = []
            import fnmatch
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, sub_type):
                    f_path_list.append(os.path.join(root, filename))
    return f_path_list, filenames


def find_corr_map(file_path_list, label_path, label_switch = ('','')):
    """
    get the label path from the image path, assume the file name is the same
    :param file_path_list: the path list of the image
    :param label_path: the path of the label folder
    :return:
    """
    fn_switch = lambda x: x.replace(label_switch[0], label_switch[1])
    return [[os.path.join(label_path, fn_switch(os.path.split(pth)[1])) for pth in file_path] for file_path in file_path_list]


def make_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist


def divide_data_set(root_path, file_name_list, ratio):
    """
    divide the dataset into root_path/train root_path/val root_path/test
    :param root_path: the root path for saving the task_dataset
    :param file_name_list: list of name of the saved file  like img1
    :param ratio: tuple of (train_ratio, val_ratio, test_ratio) from all the files
    :return:  full path of each file

    """
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    file_num = len(file_name_list)
    sub_path = {x: os.path.join(root_path, x) for x in ['train', 'val', 'test']}
    nt = [make_dir(sub_path[key]) for key in sub_path]
    if sum(nt):
        raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n"
                         "manually delete the folder to reprepare the data")
    train_num = int(train_ratio * file_num)
    val_num = int(val_ratio * file_num)
    file_name_sub_list = {}
    file_name_sub_list['train'] = file_name_list[:train_num]
    file_name_sub_list['val'] = file_name_list[train_num: train_num + val_num]
    file_name_sub_list['test'] = file_name_list[train_num + val_num:]
    saving_path_list = [os.path.join(sub_path[x], file_name + '.h5py') for x in ['train', 'val', 'test'] for file_name
                        in file_name_sub_list[x]]
    return saving_path_list


def check_same_size(img, standard):
    """
    make sure all the image are of the same size
    :param img: img to compare
    :param standard: standarded image size
    """
    assert img.shape == standard, "img size must be the same"


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
                     adaptive_padding=4):
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image,
                                                                 adaptive_padding)
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
        info['file_path'] = f['file_path'][:]
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
        asciiList = [path.encode("ascii", "ignore") for path in dic['file_path']]
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('file_path', data=asciiList, dtype=string_dt)
        f.close()
    else:
        raise ValueError('only h5py supported currently')


def save_to_h5py(path, file_list, info, file_path_list, label_list=None, verbose=True):
    """

    :param path:  path for saving file
    :param img_file_list: list of image file
    :param info: additional info
    :param img_file_path_list:  list of path/name of image file
    :param img_file_path_list:  list of path/name of corresponded label file

    :return:
    """
    dic = {'data': file_list, 'info': info, 'file_path': file_path_list, 'label': label_list}
    write_file(path, dic, type='h5py')
    if verbose:
        print('data saved: {}'.format(path))
        print(dic['info'])
        print("the shape of file{}".format(dic['data'][:].shape))
        print('the location of the file\n{}'.format(file_path_list[0]))
