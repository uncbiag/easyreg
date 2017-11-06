from os import listdir
from os.path import isfile, join
from glob import glob
import os
from scipy import misc
import numpy as np
from data.data_utils import *




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
        pair = []
        for i in range(f_num - 1):
            pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(f_num - i)]
            pair += pair_tmp
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
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(f_num - i)]
                pair += pair_tmp
        pair_list += pair
    return pair_list


def load_as_data(pair_list, type):
    '''
    input:
    pair_list is the list of the path of the paired image
    the length of the pair list is N, each unit contains pair path of the image

    return: list of the tuple of the paired numpy image
    '''

    img_pair_list = []

    for i, pair in pair_list:
        ### flatten=0 if image is required as it is
        ## flatten=1 to flatten the color layers into a single gray-scale layer
        img1 =misc.imread(pair[0],flatten=True)
        img2 =misc.imread(pair[1],flatten=True)
        img_pair_list[2*i] = (img1, img2)
        img_pair_list[2*i+1] = (img2, img1)

    assert len(img_pair_list) == 2*len(pair_list)
    height, width = img1.shape
    info = {'img_h': height, 'img_w': width, 'concat_type':type, 'img_num': len(img_pair_list)}
    return img_pair_list, info









