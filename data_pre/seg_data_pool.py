from __future__ import print_function
from data_pre.seg_data_utils import *
import random


class BaseSegDataSet(object):

    def __init__(self, file_type_list, label_switch=('', ''), dim=3):
        """
        :param name: name of data set
        :param dataset_type: ''mixed' like oasis including inter and  intra person  or 'custom' like LPBA40, only includes inter person
        :param file_type_list: the file types to be filtered, like [*1_a.bmp, *2_a.bmp]
        :param data_path: path of the dataset
        """
        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.label_path = None
        """path of the label directory"""
        self.file_name_list = []
        self.file_path_list = []
        self.file_type_list = file_type_list
        """currently only support h5py"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""
        self.dim = dim
        self.label_switch = label_switch

    def set_data_path(self, path):
        self.data_path = path

    def set_label_path(self, path):
        self.label_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)

    def set_divided_ratio(self, ratio):
        self.divided_ratio = ratio


    def gen_file_and_save_list(self):
        file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        random.shuffle(file_path_list)
        label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        file_label_path_list = [[file_path_list[idx], label_path_list[idx]] for idx in range(len(label_path_list))]
        self.num_pair = len(file_label_path_list)
        self.pair_name_list = [get_file_name(fpth) for fpth in file_path_list]
        sub_folder_dic, file_id_dic = divide_data_set(self.output_path, self.num_pair, self.divided_ratio)
        divided_path_and_name_dic = get_divided_dic(file_id_dic,file_label_path_list, self.pair_name_list)
        saving_pair_info(sub_folder_dic, divided_path_and_name_dic)



    def prepare_data(self):
        """
        preprocessing data for each dataset
        :return:
        """
        print("the output file path is: {}".format(self.output_path))
        self.gen_file_and_save_list()
        print("data preprocessing finished")





class SegDatasetPool(object):
    def create_dataset(self, dataset_name, file_type_list, label_switch=('','')):

        self.dataset_nopatched_dic = {'oai': BaseSegDataSet,
                                      'lpba': BaseSegDataSet,
                                      'ibsr': BaseSegDataSet,
                                      'cumc': BaseSegDataSet}
        dataset =self.dataset_nopatched_dic[dataset_name](file_type_list, label_switch)
        return dataset

if __name__ == "__main__":
    pass
    lpba = SegDatasetPool().create_dataset(dataset_name='lpba',file_type_list=['*.nii'])
    #data_path = "/playpen-raid/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm_hist_oasis"
    #label_path = '/playpen-raid/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    #output_path = '/playpen-raid/zyshen/data/lpba_seg'

    data_path = "/home/zyshen/proj/remote_data/LPBA40_affine_hist"
    label_path = '/home/zyshen/proj/remote_data/LPBA40_label_affine'
    output_path = '/home/zyshen/proj/local_debug/brain_seg'

    divided_ratio = (0.6, 0.2, 0.2)
    lpba.set_data_path(data_path)
    lpba.set_label_path(label_path)
    lpba.set_output_path(output_path)
    lpba.set_divided_ratio(divided_ratio)
    lpba.prepare_data()
    #
    # oai =SegDatasetPool().create_dataset(dataset_name='oai', file_type_list=['*image.nii.gz'],label_switch=('image', 'label_all'))
    # data_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled"
    # label_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled"
    # output_path = '/playpen-raid/zyshen/data/oai_seg'
    # divided_ratio = (0.6, 0.2, 0.2)
    # oai.set_data_path(data_path)
    # oai.set_label_path(data_path)
    # oai.set_output_path(output_path)
    # oai.set_divided_ratio(divided_ratio)
    # oai.prepare_data()
