from __future__ import print_function
import progressbar as pb
from multiprocessing import Pool, TimeoutError
import data_pre.module_parameters as pars

from torch.utils.data import Dataset
from copy import deepcopy
from data_pre.seg_data_utils import *
from data_pre.transform import Transform
from data_pre.partition import partition

number_of_workers = 10


class BaseGenDataSet(object):
    def __init__(self, name, file_type_list):
        """

        :param name: name of data set
        :param dataset_type: ''mixed' like oasis including inter and  intra person  or 'custom' like LPBA40, only includes inter person
        :param file_type_list: the file types to be filtered, like [*1_a.bmp, *2_a.bmp]
        :param data_path: path of the dataset
        """
        self.name = name
        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.file_name_list = []
        self.file_path_list = []
        self.file_type_list = file_type_list
        self.save_format = 'h5py'
        """currently only support h5py"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def generate_file_list(self):
        pass

    def set_data_path(self, path):
        self.data_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)

    def set_divided_ratio(self, ratio):
        self.divided_ratio = ratio

    def get_file_num(self):
        return len(self.file_path_list)

    def get_file_name_list(self):
        return self.file_name_list

    def read_file(self, file_path, is_label=False):
        """
        currently, default using file_io, reading medical format
        :param file_path:
        :param  is_label: the file_path is label_file
        :return:
        """
        # img, info = read_itk_img(file_path)
        img, info = file_io_read_img(file_path, is_label=is_label)
        return img, info

    def save_shared_info(self, info):
        save_sz_sp_to_json(info, self.output_path)

    def save_file(self):
        pass

    def prepare_data(self):
        pass


class LabeledDataSet(BaseGenDataSet):
    """
    labeled dataset  coordinate system is the same as the sitk
    """

    def __init__(self, name, file_type_list, option, label_switch=('', ''), dim=3):
        BaseGenDataSet.__init__(self, name, file_type_list)
        self.label_path = None
        self.file_path_list = []
        self.file_name_list = []
        self.dim = dim
        self.label_switch = label_switch
        self.option = option[('data_pro', {}, 'settings for data_pro')]
        self.option_trans = self.option[('transform', {}, 'settings for transform')]
        self.option_p = self.option[('partition', {}, "settings for the partition")]

        self.num_crop_per_class_per_train_img = self.option[
            ('num_crop_per_class_per_train_img', 100, 'num_crop_per_class_per_train_img')]
        self.transform_name_seq = []
        self.num_label = 0

    def set_label_path(self, path):
        self.label_path = path

    def set_label_name_switch(self, label_switch):
        self.label_switch = label_switch

    def set_transform_name_seq(self, transform_name_seq):
        self.transform_name_seq = transform_name_seq

    def get_transform_seq(self, option_trans):
        transform = Transform(self.dim, option_trans)
        return transform.get_transform_seq(self.transform_name_seq)

    def apply_transform(self, sample, transform_seq):
        for transform in transform_seq:
            sample = transform(sample)
        return sample

    def get_num_label(self):
        file_label_path_list = find_corr_map([self.file_path_list[0]], self.label_path, self.label_switch)
        label, linfo = self.read_file(file_label_path_list[0], is_label=True)
        label_list = list(np.unique(label))
        num_label = len(label_list)
        print('the num of the class: {}'.format(num_label))
        self.option_trans['shared_info']['img_size'] = list(linfo['img_size'])
        self.num_label = num_label
        self.save_shared_info(linfo)

    def train_data_processing(self, file_path_list):
        option_trans_cp = deepcopy(self.option_trans)
        option_trans_cp.print_settings_off()
        file_label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        total = len(file_path_list) * self.num_crop_per_class_per_train_img * self.num_label
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=total).start()
        count = 0
        for i, file_path in enumerate(file_path_list):
            file_name = get_file_name(file_path)
            img, info = self.read_file(file_path)
            label, linfo = self.read_file(file_label_path_list[i], is_label=True)
            label_list = list(np.unique(label))
            label_density = np.bincount(label.reshape(-1).astype(np.int32)) / len(label.reshape(-1))
            option_trans_cp['shared_info']['label_list'] = label_list
            option_trans_cp['shared_info']['label_density'] = label_density
            num_label = len(label_list)
            if self.num_label != num_label:  # s37 in lpba40 has one more label than others
                print("Warnning!!!!, The num of classes {} are not the same in file{}".format(num_label, file_path))
            transform_seq = self.get_transform_seq(option_trans_cp)
            sample = {'image': np_to_sitk(img, info), 'seg': np_to_sitk(label, info)}
            for _ in range(self.num_label):
                for _ in range(self.num_crop_per_class_per_train_img):
                    patch_transformed = self.apply_transform(sample, transform_seq)
                    saving_patch_per_img(patch_transformed, self.saving_path_dic[file_name])
                    count += 1
                    pbar.update(count)
        pbar.finish()

    def test_data_processing(self, file_path_list):
        partition_ins = partition(self.option_p)
        file_label_path_list = find_corr_map(file_path_list, self.label_path, self.label_switch)
        for i, file_path in enumerate(file_path_list):
            file_name = get_file_name(file_path)
            img, info = self.read_file(file_path)
            label, linfo = self.read_file(file_label_path_list[i], is_label=True)
            sample = {'image': np_to_sitk(img, info), 'seg': np_to_sitk(label, info)}
            patches = partition_ins(sample)
            saving_patches_per_img(patches, self.saving_path_dic[file_name])

    def get_file_list(self):
        self.file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        random.shuffle(self.file_path_list)
        self.file_name_list = [os.path.split(file_path)[1].split('.')[0] for file_path in self.file_path_list]

    def get_save_path_list(self):
        self.saving_path_dic, self.file_path_dic = divide_data_set(self.output_path, self.file_path_list,
                                                                   self.divided_ratio)

    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        print("starting preapare data..........")
        print("the output file path is: {}".format(self.output_path))
        self.get_file_list()
        self.get_save_path_list()
        self.get_num_label()
        file_patitions = np.array_split(self.file_path_dic['train'], number_of_workers)
        with Pool(processes=number_of_workers) as pool:
            res = pool.map(self.train_data_processing, file_patitions)
        file_patitions = np.array_split(self.file_path_dic['val'] + self.file_path_dic['test'], number_of_workers)
        with Pool(processes=number_of_workers) as pool:
            res = pool.map(self.test_data_processing, file_patitions)
        print("data preprocessing finished")


class OAIDataSet(LabeledDataSet):
    def __init__(self, name, option):
        LabeledDataSet.__init__(self, name, ['*.nii'], option, label_switch=('image', 'label_all'))


class LPBADataSet(LabeledDataSet):
    def __init__(self, name, option):
        LabeledDataSet.__init__(self, name, ['*.nii'], option)


class IBSRDataSet(LabeledDataSet):
    def __init__(self, name, option):
        LabeledDataSet.__init__(self, name, ['*.nii'], option)


class CUMCDataSet(LabeledDataSet):
    def __init__(self, name, option):
        LabeledDataSet.__init__(self, name, ['*.nii'], option)


if __name__ == "__main__":
    pass

    # #########################       OASIS TESTING           ###################################3
    #
    # path = '/playpen/zyshen/data/oasis'
    # name = 'oasis'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #oasis  intra testing
    # full_comb = True
    # sched= 'intra'
    #
    # output_path = '/playpen/zyshen/data/'+ name+'_pre_'+ sched
    # oasis = Oasis2DDataSet(name='oasis',sched=sched, full_comb=True)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()


    # ###################################################
    # # oasis inter testing
    # sched='inter'
    # full_comb = False
    # output_path = '/playpen/zyshen/data/' + name + '_pre_' + sched
    # oasis = Oasis2DDataSet(name='oasis', sched=sched, full_comb=full_comb)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()




    ###########################       LPBA TESTING           ###################################
    path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    file_type_list = ['*.nii']
    full_comb = False
    name = 'lpba'
    output_path = '/playpen/zyshen/data/' + name + '_pre'
    divided_ratio = (0.6, 0.2, 0.2)

    ###################################################
    # lpba testing

    option = pars.ParameterDict()

    lpba = LPBADataSet(name=name, option=option)
    lpba.set_data_path(path)
    lpba.set_output_path(output_path)
    lpba.set_divided_ratio(divided_ratio)
    lpba.set_label_path(label_path)
    lpba.prepare_data()


    # ###########################       LPBA Slicing TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre_slicing'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    #
    # lpba = LPBADataSet(name=name, full_comb=full_comb)
    # lpba.set_slicing(90,1)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()





