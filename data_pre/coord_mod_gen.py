from data_pre.seg_data_utils import *
import os
import numpy as np
class MultiTxtGen(object):
    def __init__(self, mode_list, data_path, img_type='.nii'):
        self.mode_list = mode_list
        self.data_path = data_path
        self.img_type = img_type
        self.file_path_list = []
        self.ratio = (0.95, 0.05)
        self.file_num = -1
        self.divide_ind={}

    def set_divide_ratio(self,ratio):
        self.ratio =  ratio


    def get_file_list(self):
        mode_0 = self.mode_list[0]
        img_type = ['*'+ mode_0+ self.img_type]
        self.file_path_list=get_file_path_list(self.data_path,img_type)

    def identity_map(sz, dtype='float32'):
        """
        Returns an identity map.

        :param sz: just the spatial dimensions, i.e., XxYxZ
        :param spacing: list with spacing information [sx,sy,sz]
        :param dtype: numpy data-type ('float32', 'float64', ...)
        :return: returns the identity map of dimension dimxXxYxZ
        """

        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.array(id.astype(dtype))

        return id



    def gen_coord_map(self):
        for path in self.file_path_list:
            image = sitk.ReadImage(path)
            numpy_shape = sitk.GetArrayFromImage(image).shape
            grid = np.zeros(numpy_shape, dtype=np.float32)
            grid[0, ...] = np.expand_dims(
                np.repeat(np.expand_dims(np.arange(0, 1, 1/ numpy_shape[0])[:numpy_shape[0]], 0), repeats=numpy_shape[1], axis=0).T, 0)
            grid[1, ...] = np.expand_dims(
                np.repeat(np.expand_dims(np.arange(0, 1, 1/ width)[:width], 0), repeats=height, axis=0), 0)

    def gen_divide_index(self):
        ind = list(range(self.file_num))
        import random
        random.shuffle(ind)
        train_num = int(self.ratio[0] * self.file_num)
        val_num = int(self.ratio[1] * self.file_num)
        test_num = self.file_num - train_num - val_num
        self.divide_ind['train'] = ind[:train_num]
        self.divide_ind['val'] = ind[train_num:train_num + val_num]
        self.divide_ind['test'] = ind[train_num + val_num:]


    def write_into_txt(self):
        for sess in self.sesses:
            with open(os.path.join(self.output_path,sess+'.txt'), 'w') as f:
                for ind in self.divide_ind[sess]:
                    mode_list = [self.file_path_list_dic[self.mode_list[0]][ind],self.file_path_list_dic['label'][ind]]
                    concat_info = str_concat(mode_list,linker=',')
                    f.write(concat_info+'\n')

    def gen_text(self):
        self.get_file_list()
        self.get_label_list()
        self.gen_divide_index()
        self.write_into_txt()






#
# # Brat dataset
# model_list = ['flair','t1','t1ce','t2']
# data_path = '/playpen/zyshen/data/MICCAI_BraTS17_Data_Training'
# output_path = '/playpen/zyshen/data/MICCAI_BraTS17_Data_Training/debug'
# brat = MultiTxtGen(mode_list=model_list, data_path=data_path, output_path=output_path, img_type='.nii.gz',label_replace='seg')
# brat.gen_text()
#
