from os import listdir
from os.path import isfile, join
from glob import glob
import os
from scipy import misc
import numpy as np
import h5py
from data.data_utils import *
import SimpleITK as sitk




def list_dic(path):
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]



def list_pairwise(path, img_type, skip, sched):
    '''
     return the list of  paths of the paired image  [N,2]
    skip: if skip then sampling can return the pairs of the image of mri from different time'''
    dic_list= list_dic(path)
    if sched == 'intra':
        pair_list = intra_pair(path, dic_list, img_type, skip)
    else:
        pair_list = inter_pair(path, dic_list, img_type)
    return pair_list



def inter_pair(path, dic_list, type):
    # type here should be [*1_a.bmp, *2_a.bmp]
    pair_list=[]
    for sub_type in type:
        f_path = join(path,'**', sub_type)
        f_filter = glob(f_path, recursive=True)
        f_num = len(f_filter)
        pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        pair_list += pair
    return pair_list





def intra_pair(path, dic_list, type, skip):
    # type here can simply be *.bmp
    pair_list = []
    for dic in dic_list:
        f_path = join(path, dic, type[0])
        f_filter = glob(f_path)
        f_num = len(f_filter)
        assert f_num != 0
        if not skip:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    return pair_list


def load_as_data(pair_list):
    '''
    input:
    pair_list is the list of the path of the paired image
    the length of the pair list is N, each unit contains pair path of the image

    return: list of the tuple of the paired numpy image
    '''

    img_pair_list = []
    standard=()

    for i, pair in enumerate(pair_list):
        ### flatten=0 if image is required as it is
        ## flatten=1 to flatten the color layers into a single gray-scale layer
        img1 =read_img(pair[0])
        img2 =read_img(pair[1])

        # check img size
        if i==0:
            standard = img1.shape
            check_same_size(img2, standard)
        else:
            check_same_size(img1,standard)
            check_same_size(img2,standard)
        normalize_img(img1)
        normalize_img(img2)
        img_pair_list += [(img1, img2)]
        img_pair_list += [(img2, img1)]

    assert len(img_pair_list) == 2*len(pair_list)
    info = {'img_h': standard[0], 'img_w': standard[1], 'pair_num': len(img_pair_list)}
    return img_pair_list, info


def check_same_size(img, standard):
    assert img.shape == standard, "img size must be the same"

def normalize_img(image):
    image[:] = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1


def read_img(path):
    itkimage = sitk.ReadImage(path)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return np.squeeze(ct_scan)



def read_file(path, type='h5py'):
    if type == 'h5py':
        f = h5py.File(path, 'r')
        data = f['data'][:]
        info = {}
        for key in f.attrs:
            info[key]= f.attrs[key]
        f.close()
        return {'data':data, 'info': info}


def write_file(path, dic, type='h5py'):
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.create_dataset('data',data=dic['data'])
        for key, value in dic['info'].items():
            f.attrs[key] = value
        f.close()



def save_to_h5py(path, img_pair_list, info):
    dic = {'data': img_pair_list, 'info': info}
    write_file(path, dic, type='h5py')



