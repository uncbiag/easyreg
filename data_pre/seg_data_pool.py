from __future__ import print_function
from data_pre.seg_data_utils import *
import random
from multiprocessing import *
from tools.image_rescale import resize_input_img_and_save_it_as_tmp
from functools import partial

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
        number_of_workers = 6
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
                print("warning label: is not in standard label index, and would be convert to 0".format(l_id))
            label_map[np.where(label_map == l_id)] = st_index
        return label_map

    def _filter_and_save_label(self, label_path_list, saving_path=None):

        for label_path in label_path_list:
            label_sitk = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(label_sitk)
            label_np = label_np.astype(np.float32)
            label_filtered = self.convert_to_standard_label_map(label_np,label_path)
            label_filtered_sitk = sitk.GetImageFromArray(label_filtered)
            label_filtered_sitk.SetSpacing(label_sitk.GetSpacing())
            label_filtered_sitk.SetOrigin(label_sitk.GetOrigin())
            label_filtered_sitk.SetDirection(label_sitk.GetDirection())
            fname = os.path.split(label_path)[-1]
            file_saving_path = os.path.join(saving_path, fname)
            sitk.WriteImage(label_filtered_sitk,file_saving_path)


    def filter_and_save_label(self, label_path_list):
        number_of_workers = 6
        file_patitions = np.array_split(label_path_list, number_of_workers)
        saving_path = os.path.join(self.output_path,"label_filtered")
        os.makedirs(saving_path,exist_ok=True)
        filter_label_func = partial(self._filter_and_save_label, saving_path=saving_path)
        with Pool(processes=number_of_workers) as pool:
            pool.map(filter_label_func, file_patitions)
        self.label_path = saving_path



    def _resize_img_label(self, idx_list, file_path_list, label_path_list, saving_path_img,saving_path_label):
        for i in idx_list:
            fname = get_file_name(file_path_list[i]) + '.nii.gz'
            resize_input_img_and_save_it_as_tmp(file_path_list[i], is_label=False, keep_physical=True, fname=fname,
                                                saving_path=saving_path_img, fixed_sz=self.img_after_resize)
            fname = get_file_name(label_path_list[i]) + '.nii.gz'
            resize_input_img_and_save_it_as_tmp(label_path_list[i], is_label=True, keep_physical=True, fname=fname,
                                                saving_path=saving_path_label, fixed_sz=self.img_after_resize)



    def resize_img_label(self, file_path_list, label_path_list):
        if self.img_after_resize is not None:
            saving_path_img = os.path.join(self.output_path,'resized_img')
            saving_path_label = os.path.join(self.output_path,"resized_label")
            os.makedirs(saving_path_img,exist_ok=True)
            os.makedirs(saving_path_label, exist_ok=True)
            number_of_workers = 6
            idx_partitions = np.array_split(list(range(len(file_path_list))), number_of_workers)
            resize_func = partial(self._resize_img_label, file_path_list=file_path_list, label_path_list=label_path_list, saving_path_img=saving_path_img, saving_path_label=saving_path_label)

            with Pool(processes=number_of_workers) as pool:
                pool.map(resize_func, idx_partitions)

            self.data_path = saving_path_img
            self.label_path = saving_path_label










    def data_preprocess(self):
        file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        #random.shuffle(file_path_list)
        label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        self.resize_img_label(file_path_list,label_path_list)
        file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        self.get_shared_label_index(label_path_list)
        self.filter_and_save_label(label_path_list)
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
    pass
    # lpba = SegDatasetPool().create_dataset(dataset_name='lpba',file_type_list=['*.nii','*nii.gz'])
    # data_path = "/playpen-raid/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm_hist_oasis"
    # label_path = '/playpen-raid/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # output_path = '/playpen-raid/zyshen/data/lpba_seg_resize'
    #
    # # data_path = "/home/zyshen/proj/remote_data/LPBA40_affine_hist"
    # # label_path = '/home/zyshen/proj/remote_data/LPBA40_label_affine'
    # # output_path = '/home/zyshen/proj/local_debug/brain_seg'
    #
    #
    # divided_ratio = (0.625, 0.125, 0.25)
    # lpba.set_data_path(data_path)
    # lpba.set_label_path(label_path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.img_after_resize =(196,164,196)
    # lpba.prepare_data()

    oai = SegDatasetPool().create_dataset(dataset_name='oai', file_type_list=['*image.nii.gz'])

    label_switch = ('image', 'masks')

    data_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    label_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    output_path = '/playpen-raid/zyshen/data/oai_seg'
    divided_ratio = (0.8, 0.1, 0.1)
    oai.label_switch = label_switch
    oai.set_data_path(data_path)
    oai.set_label_path(label_path)
    oai.set_output_path(output_path)
    oai.set_divided_ratio(divided_ratio)
    oai.img_after_resize = ( 160,200,200)
    oai.prepare_data()
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
