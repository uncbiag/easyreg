from __future__ import print_function
import os, sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
from data_pre.seg_data_utils import *
import random
from multiprocessing import *
from tools.image_rescale import resize_input_img_and_save_it_as_tmp
from functools import partial

num_c = 10
number_of_workers= 6

class BaseSegDataSet(object):

    def __init__(self, file_type_list, label_switch=('', ''), filter_label=True,dim=3):
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
        self.filter_label = filter_label
        self.find_corr_label = find_corr_map
        self.get_file_name  = get_file_name
        self.file_name_list = []
        self.file_path_list = []
        self.file_type_list = file_type_list
        self.max_used_train_samples = -1
        """currently only support h5py"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""
        self.dim = dim
        self.label_switch = label_switch
        self.sever_switch = None
        self.standard_label_index = []
        self.img_after_resize = None



    def set_data_path(self, path):
        self.data_path = path

    def set_label_path(self, path):
        self.label_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)

    def set_divided_ratio(self, ratio):
        self.divided_ratio = ratio


    def _get_label_index_per_img(self,label_path_list):
        label_index_list = []
        for path in label_path_list:
            label_sitk = sitk.ReadImage(path)
            label_np = sitk.GetArrayFromImage(label_sitk)
            label_index = np.unique(label_np)
            label_index_list.append(list(label_index))
        return label_index_list

    def get_shared_label_index(self, label_path_list):
        file_patitions = np.array_split(label_path_list, number_of_workers)
        with Pool(processes=number_of_workers) as pool:
            label_index_sub_list = pool.map(self._get_label_index_per_img, file_patitions)
        label_index_list = [label_index for sub_list in label_index_sub_list for label_index in sub_list]
        shared_label_set = dict()
        for i,label_index in enumerate(label_index_list):
            label_set = set(label_index)
            if i == 0:
                shared_label_set = label_set
            else:
                shared_label_set = shared_label_set.intersection(label_set)
        shared_label_list = list(shared_label_set)
        print("the shared label list is {}".format(shared_label_list))
        self.standard_label_index = shared_label_list
        self.num_label = len(self.standard_label_index)

    def convert_to_standard_label_map(self, label_map, file_path):
        cur_label_list = list(np.unique(label_map))
        num_label = len(cur_label_list)
        if self.num_label != num_label:  # s37 in lpba40 has one more label than others
            print("Warnning!!!!, The num of classes {} are not the same in file{}".format(num_label, file_path))

        for l_id in cur_label_list:
            if l_id in self.standard_label_index:
                st_index = self.standard_label_index.index(l_id)
            else:
                # assume background label is 0
                st_index = 0
                print("warning label:{} is not in standard label index, and would be convert to 0".format(l_id))
            label_map[np.where(label_map == l_id)] = st_index
        return label_map

    def _filter_and_save_label(self, label_path_list, saving_path=None):

        for label_path in label_path_list:
            label_sitk = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(label_sitk)
            label_np = label_np.astype(np.int32)
            label_filtered = self.convert_to_standard_label_map(label_np,label_path)
            label_filtered_sitk = sitk.GetImageFromArray(label_filtered)
            label_filtered_sitk.SetSpacing(label_sitk.GetSpacing())
            label_filtered_sitk.SetOrigin(label_sitk.GetOrigin())
            label_filtered_sitk.SetDirection(label_sitk.GetDirection())
            fname = get_file_name(label_path)
            file_saving_path = os.path.join(saving_path, fname+".nii.gz")
            sitk.WriteImage(label_filtered_sitk,file_saving_path)


    def filter_and_save_label(self, label_path_list):
        file_patitions = np.array_split(label_path_list, number_of_workers)
        saving_path = os.path.join(self.output_path,"label_filtered")
        os.makedirs(saving_path,exist_ok=True)
        filter_label_func = partial(self._filter_and_save_label, saving_path=saving_path)
        with Pool(processes=number_of_workers) as pool:
            pool.map(filter_label_func, file_patitions)
        self.label_path = saving_path



    def _resize_img_label(self, idx_list, file_path_list, label_path_list, saving_path_img,saving_path_label):
        for i in idx_list:
            fname = self.get_file_name(file_path_list[i]) + '.nii.gz'
            resize_input_img_and_save_it_as_tmp(file_path_list[i], is_label=False, keep_physical=True, fname=fname,
                                                saving_path=saving_path_img, fixed_sz=self.img_after_resize)
            resize_input_img_and_save_it_as_tmp(label_path_list[i], is_label=True, keep_physical=True, fname=fname,
                                                saving_path=saving_path_label, fixed_sz=self.img_after_resize)



    def resize_img_label(self, file_path_list, label_path_list):
        if self.img_after_resize is not None:
            saving_path_img = os.path.join(self.output_path,'resized_img')
            saving_path_label = os.path.join(self.output_path,"resized_label")
            os.makedirs(saving_path_img,exist_ok=True)
            os.makedirs(saving_path_label, exist_ok=True)
            idx_partitions = np.array_split(list(range(len(file_path_list))), number_of_workers)
            resize_func = partial(self._resize_img_label, file_path_list=file_path_list, label_path_list=label_path_list, saving_path_img=saving_path_img, saving_path_label=saving_path_label)

            with Pool(processes=number_of_workers) as pool:
                pool.map(resize_func, idx_partitions)

            self.data_path = saving_path_img
            self.label_path = saving_path_label
            self.get_file_name = get_file_name
            file_path_list = get_file_path_list(self.data_path, ["*.nii.gz"])
            label_path_list = find_corr_map(file_path_list, self.label_path)
        return file_path_list, label_path_list










    def data_preprocess(self):
        file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        #random.shuffle(file_path_list)
        label_path_list = self.find_corr_label(file_path_list, self.label_path, self.label_switch)
        file_path_list, label_path_list = self.resize_img_label(file_path_list,label_path_list)
        if self.filter_label:
            self.get_shared_label_index(label_path_list)
            self.filter_and_save_label(label_path_list)
        label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        if self.sever_switch is None:
            file_label_path_list = [[file_path_list[idx], label_path_list[idx]] for idx in range(len(label_path_list))]
        else:
            file_label_path_list = [[file_path_list[idx].replace(self.sever_switch[0],self.sever_switch[1])
                                        , label_path_list[idx].replace(self.sever_switch[0], self.sever_switch[1])] for idx in range(len(label_path_list))]

        self.num_pair = len(file_label_path_list)
        self.pair_name_list = [self.get_file_name(fpth) for fpth in file_path_list]
        sub_folder_dic, file_id_dic = divide_data_set(self.output_path, self.num_pair, self.divided_ratio)
        divided_path_and_name_dic = get_divided_dic(file_id_dic,file_label_path_list, self.pair_name_list)
        if self.max_used_train_samples>0:
            divided_path_and_name_dic['file_path_list']['train']=divided_path_and_name_dic['file_path_list']['train'][:self.max_used_train_samples]#  should be -1 in most cases
            divided_path_and_name_dic['file_name_list']['train']=divided_path_and_name_dic['file_name_list']['train'][:self.max_used_train_samples]
        print("num of train samples is {}".format(len(divided_path_and_name_dic['file_name_list']['train'])))
        saving_pair_info(sub_folder_dic, divided_path_and_name_dic)




    def prepare_data(self):
        """
        preprocessing data for each dataset
        :return:
        """
        print("the output file path is: {}".format(self.output_path))
        self.data_preprocess()
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
    data_path = "../demo/lpba_examples/data"
    label_path = "../demo/lpba_examples/label"
    divided_ratio = (0.4, 0.4, 0.2)  # ratio for train val test
    output_path = '/playpen-raid1/zyshen/debug/lpba_seg_split'
    lpba = SegDatasetPool().create_dataset(dataset_name='lpba', file_type_list=['*nii.gz'])
    lpba.set_data_path(data_path)
    lpba.set_label_path(label_path)
    lpba.set_output_path(output_path)
    lpba.set_divided_ratio(divided_ratio)
    lpba.prepare_data()

    # num_c_list = [5, 10, 15, 20, 25]
    #
    # for num_c in num_c_list:
    #     lpba = SegDatasetPool().create_dataset(dataset_name='lpba',file_type_list=['*nii.gz'])
    #
    #     label_switch = ('_image.nii.gz', '_label.nii.gz')
    #     sever_switch = ('/playpen-raid', '/pine/scr/z/y')
    #     #sever_switch = None
    #     data_path = "/playpen-raid/zyshen/data/lpba_seg_resize/resized_img"
    #     label_path = '/playpen-raid/zyshen/data/lpba_seg_resize/label_filtered'
    #
    #     output_path = "/playpen-raid/zyshen/data/lpba_seg_resize/baseline_sever/{}case".format(num_c)
    #     divided_ratio =  (0.625, 0.125, 0.25)
    #     lpba.label_switch = label_switch
    #     lpba.sever_switch = sever_switch
    #     lpba.max_used_train_samples=num_c
    #     lpba.set_data_path(data_path)
    #     lpba.set_label_path(label_path)
    #     lpba.set_output_path(output_path)
    #     lpba.set_divided_ratio(divided_ratio)
    #     lpba.img_after_resize = (196, 164, 196)
    #     lpba.prepare_data()

    #
    # num_c_list = [5, 10, 15, 20, 25]
    # sever_on = True
    # cur_path = '/pine/scr/z/y' if sever_on else '/playpen-raid'
    #
    # for num_c in num_c_list:
    #     lpba = SegDatasetPool().create_dataset(dataset_name='lpba',file_type_list=['*_image.nii.gz'])
    #
    #     label_switch = ('_image.nii.gz', '_label.nii.gz')
    #     sever_switch = (cur_path, '/pine/scr/z/y')
    #     # sever_switch = None
    #
    #     data_path = "{}/zyshen/data/lpba_seg_resize/baseline/aug/gen_lresol_atlas/{}case".format(cur_path,num_c)
    #     label_path = data_path
    #     output_path = "{}/zyshen/data/lpba_seg_resize/baseline/aug/sever/gen_lresol_atlas/{}case".format(cur_path,num_c)
    #     divided_ratio = (1, 0, 0)
    #     lpba.label_switch = label_switch
    #     lpba.sever_switch = sever_switch if sever_on else None
    #     lpba.set_data_path(data_path)
    #     lpba.set_label_path(label_path)
    #     lpba.set_output_path(output_path)
    #     lpba.set_divided_ratio(divided_ratio)
    #     lpba.img_after_resize = (196, 164, 196)
    #     lpba.prepare_data()
    #     import subprocess
    #     cmd = "\n cp -r {}/zyshen/data/lpba_seg_resize/baseline/{}case/val ".format(cur_path,num_c) +output_path
    #     cmd += "\n cp -r {}/zyshen/data/lpba_seg_resize/baseline/{}case/test ".format(cur_path,num_c) +output_path
    #     cmd += "\n cp -r {}/zyshen/data/lpba_seg_resize/baseline/{}case/debug ".format(cur_path,num_c) +output_path
    #     process = subprocess.Popen(cmd, shell=True)
    #     process.wait()
    # num_c_list= [5,10,15,20,25]
    #
    # for num_c in num_c_list:
    #     lpba = SegDatasetPool().create_dataset(dataset_name='lpba', file_type_list=['*_image.nii.gz'])
    #
    #     label_switch = ('_image.nii.gz', '_label.nii.gz')
    #     sever_switch=('/playpen-raid','/pine/scr/z/y')
    #     #sever_switch = None
    #
    #
    #     data_path = "/playpen-raid/zyshen/data/lpba_reg/train_with_{}/lpba_ncc_reg1/gen_lresol".format(num_c)
    #     label_path = data_path
    #     output_path = data_path + '_seg_sever'
    #     divided_ratio = (1, 0, 0)
    #     lpba.label_switch = label_switch
    #     lpba.sever_switch = sever_switch
    #     lpba.set_data_path(data_path)
    #     lpba.set_label_path(label_path)
    #     lpba.set_output_path(output_path)
    #     lpba.set_divided_ratio(divided_ratio)
    #     lpba.img_after_resize = (196,164,196)
    #     lpba.prepare_data()





    # oai = SegDatasetPool().create_dataset(dataset_name='oai', file_type_list=['*image.nii.gz'])
    #
    # label_switch = ('image', 'masks')
    #
    # data_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    # label_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    # output_path = '/playpen-raid/zyshen/data/oai_seg'
    # divided_ratio = (0.8, 0.1, 0.1)
    # oai.label_switch = label_switch
    # oai.set_data_path(data_path)
    # oai.set_label_path(label_path)
    # oai.set_output_path(output_path)
    # oai.set_divided_ratio(divided_ratio)
    # oai.img_after_resize = ( 160,200,200)
    # oai.prepare_data()

    # oai = SegDatasetPool().create_dataset(dataset_name='oai', file_type_list=['*image.nii.gz'])
    #
    # label_switch = ('image', 'masks')
    # sever_switch = ('/playpen-raid/olut', '/pine/scr/z/y/zyshen/data')
    #
    # data_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    # label_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    # output_path = '/playpen-raid/zyshen/data/oai_seg/baseline_sever/{}case'.format(num_c)
    # divided_ratio = (0.8, 0.1, 0.1)
    # oai.label_switch = label_switch
    # oai.sever_switch = sever_switch
    # oai.set_data_path(data_path)
    # oai.set_label_path(label_path)
    # oai.set_output_path(output_path)
    # oai.set_divided_ratio(divided_ratio)
    # oai.img_after_resize = ( 160,200,200)
    # oai.prepare_data()




    # num_c_list = [10, 20, 30, 40,60,80,100]
    # sever_on = True
    # cur_path = '/pine/scr/z/y' if sever_on else '/playpen-raid'
    # for num_c in num_c_list:
    #     oai = SegDatasetPool().create_dataset(dataset_name='oai', file_type_list=['*_image.nii.gz'])
    #
    #     label_switch = ('_image.nii.gz', '_label.nii.gz')
    #     sever_switch=('/playpen-raid','/pine/scr/z/y')
    #     #sever_switch=None
    #
    #     # data_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    #     # label_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    #     data_path = "{}/zyshen/data/oai_seg/baseline/aug/gen_lresol_atlas/{}case".format(cur_path,num_c)
    #     label_path = data_path
    #     output_path = "{}/zyshen/data/oai_seg/baseline/aug/sever/gen_lresol_atlas/{}case".format(cur_path,num_c)
    #     divided_ratio = (1,0,0)
    #     oai.label_switch = label_switch
    #     oai.sever_switch = sever_switch if sever_on else None
    #     oai.set_data_path(data_path)
    #     oai.set_label_path(label_path)
    #     oai.set_output_path(output_path)
    #     oai.set_divided_ratio(divided_ratio)
    #     oai.img_after_resize = (160,200,200)
    #     oai.prepare_data()
    #     import subprocess
    #
    #     cmd = "\n cp -r {}/zyshen/data/oai_seg/baseline/{}case/val ".format(cur_path,num_c) + output_path
    #     cmd += "\n cp -r {}/zyshen/data/oai_seg/baseline/{}case/test ".format(cur_path,num_c) + output_path
    #     cmd += "\n cp -r {}/zyshen/data/oai_seg/baseline/{}case/debug ".format(cur_path,num_c) + output_path
    #     process = subprocess.Popen(cmd, shell=True)
    #     process.wait()

