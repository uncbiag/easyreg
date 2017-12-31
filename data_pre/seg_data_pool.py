from __future__ import print_function
import progressbar as pb

from torch.utils.data import Dataset

from data_pre.seg_data_utils import *
from data_pre.transform import  Transform



class BaseSegDataSet(object):

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


    def set_divided_ratio(self,ratio):
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


    def save_shared_info(self,info):
        save_sz_sp_to_json(info, self.output_path)

    def save_file(self):
        pass


    def prepare_data(self):
        pass





class LabeledDataSet(BaseSegDataSet):
    """
    labeled dataset
    """
    def __init__(self, name, file_type_list):
        BaseSegDataSet.__init__(self, name, file_type_list)
        self.label_path = None
        self.file_label_path_list=[]
        self.label_switch = ('','')



    def set_label_path(self, path):
        self.label_path = path

    def set_label_name_switch(self,label_switch):
        self.label_switch = label_switch

    def apply_transform(self):
        tranform = Transform()


    def crop_into_patches(self):
        pass




    def save_file_to_file(self):
        """
        save the file into h5py
        :param file_path_list: N*1  [[full_path_img1],[full_path_img2]]
        :param file_name_list: N*1 : [fileName1, .....]
        :param ratio:  divide dataset into training val and test, based on ratio, e.g [0.7, 0.1, 0.2]
        :param saving_path_list: N*1 list of path for output files e.g [output_path/train/filename.h5py,.........]
        :param info: dic including file information
        :param normalized_sched: normalized the image
        """
        random.shuffle(self.file_path_list)
        self.file_label_path_list = find_corr_map(self.file_path_list, self.label_path, self.label_switch)
        saving_path_list = divide_data_set(self.output_path, self.file_name_list, self.divided_ratio)
        img_size = ()
        info = None
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(self.file_path_list)).start()
        for i, file in enumerate(self.file_path_list):
            img, info = self.read_file(file)
            label, linfo = self.read_file(self.file_label_path_list[i][0], is_label=True)
            if i == 0:
                img_size = img.shape
                check_same_size(label, img_size)
            else:
                check_same_size(img, img_size)
                check_same_size(label, img_size)
                # Normalized has been done in fileio, though additonal normalization can be done here
                # normalize_img(img1, self.normalize_sched)
                # normalize_img(img2, self.normalize_sched)
            img_file = np.asarray([img])
            label_file = np.asarray([label])
            info = info
            save_to_h5py(saving_path_list[i], img_file, info, [self.file_name_list[i]], label_file, verbose=False)
            pbar.update(i + 1)
        pbar.finish()
        self.save_shared_info(info)

    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        print("starting preapare data..........")
        print("the output file path is: {}".format(self.output_path))
        self.file_path_list, self.file_name_list = get_filename_list(self.data_path, self.file_type_list)
        print("the total num of file is {}".format(self.get_file_num()))
        self.save_file()
        print("data preprocessing finished")











class LPBADataSet(LabeledDataSet):
    def __init__(self, name):
        LabeledDataSet.__init__(self, name, ['*.nii'])




class IBSRDataSet(LabeledDataSet):
    def __init__(self, name):
        LabeledDataSet.__init__(self, name, ['*.nii'])



class CUMCDataSet(LabeledDataSet):
    def __init__(self, name):
        LabeledDataSet.__init__(self, name, ['*.nii'])




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




    # ###########################       LPBA TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # file_type_list = ['*.nii']
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    # sched= 'intra'
    #
    # lpba = LPBADataSet(name=name, full_comb=full_comb)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()


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





