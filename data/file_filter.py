from os import listdir
from os.path import isfile, join
from glob import glob
import os
from scipy import misc
import numpy as np
from data.data_utils import *




def list_dic(path):
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]



def list_pairwise(path, type, skip):
    '''
     return the list of  paths of the paired image  [N,2]
    skip: if skip then sampling can return the pairs of the image of mri from different time'''
    dic_list= list_dic(path)
    pair_list =[]
    for dic in dic_list:
        f_path = join(path,dic,type)
        f_filter = glob(f_path)
        f_num = f_filter
        assert f_num !=0
        if not skip:
            pair = [[f_filter[idx], f_filter[idx+1]] for idx in range(f_num-1)]
        else:
            pair =[]
            for i in range(f_num-1):
                pair_tmp = [[f_filter[i], f_filter[idx+i]] for idx in range(f_num-i)]
                pair += pair_tmp
        pair_list += pair
    return pair_list



def load_as_data(pair_list, type):
    '''
    input:
    pair_list is the list of the path of the paired image
    the length of the pair list is N, each unit contains pair path of the image
    type: depth concat,  width concat

    return: list of the numpy image
    '''
    img_pair_list = []
    for i, pair in pair_list:
        ### flatten=0 if image is required as it is
        ## flatten=1 to flatten the color layers into a single gray-scale layer
        img1 =misc.imread(pair[0],flatten=True)
        img2 =misc.imread(pair[1],flatten=True)
        if type == 'depth_concat':
            img_pair_list[2*i] =  np.stack([img1,img2],axis=0)
            img_pair_list[2*i+1] = np.stack([img2, img1], axis=0)
        elif type == 'width_concat':
            img_pair_list[2*i] = np.concatenate((img1,img2),axis=1)
            img_pair_list[2*i+1] = np.concatenate((img2,img1),axis=1)


    assert len(img_pair_list) == 2*len(pair_list)
    height, width = img1.shape
    info = {'img_h': height, 'img_w': width, 'concat_type':type, 'img_num': len(img_pair_list)}
    return img_pair_list, info



def save_to_h5py(path, img_pair_list, info):
    dic = {'data': img_pair_list, 'info': info}
    write_file(path, dic, type='h5py')





