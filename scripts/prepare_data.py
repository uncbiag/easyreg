from data.data_utils import *
from data.file_filter import *
from os.path import join

def prepare_data(save_path, img_type, path='./data',skip=True, sched='intra'):
    '''
    default:
    path: './data'
    img_type: '*.bmp'
    skip: True
    concat_type: depth_concat

     '''

    pair_list = list_pairwise(path, img_type, skip,sched)
    img_pair_list, info = load_as_data(pair_list)
    save_to_h5py(save_path, img_pair_list, info)



if __name__ == "__main__":

    path = './data'
    img_type = ['*.bmp']
    skip = True
    sched = 'intra'
    save_path = './data_'+ sched +'.h5py'


    prepare_data(save_path, img_type, path, skip, sched)
