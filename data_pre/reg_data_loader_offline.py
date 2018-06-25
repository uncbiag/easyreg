from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_pre.reg_data_utils import *
from time import time


class RegistrationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path,phase=None, transform=None, seg_option=None, reg_option=None):
        """

        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.data_type = '*.nii.gz'
        self.get_file_list()
        self.resize = True
        self.reg_option = reg_option
        self.resize_factor=reg_option['resize_factor']

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        self.path_list = read_txt_into_list(os.path.join(self.data_path,'pair_path_list.txt'))
        self.name_list = read_txt_into_list(os.path.join(self.data_path, 'pair_name_list.txt'))
        if len(self.name_list)==0:
            self.name_list = ['pair_{}'.format(idx) for idx in range(len(self.path_list))]


    def resize_img(self, img, is_label=False):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = self.resize_factor
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [img_sz[i]*factor[i] for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        return img_resampled


    def __read_and_clean_itk_info(self,path):
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))



    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        pair_path = self.path_list[idx]
        filename = self.name_list[idx]
        sitk_pair_list = [ self.__read_and_clean_itk_info(pt) for pt in pair_path]
        if self.resize:
            sitk_pair_list[0] = self.resize_img(sitk_pair_list[0])
            sitk_pair_list[1] = self.resize_img(sitk_pair_list[1])
            if len(pair_path)==4:
                sitk_pair_list[2] = self.resize_img(sitk_pair_list[2], is_label=True)
                sitk_pair_list[3] = self.resize_img(sitk_pair_list[3], is_label=True)

        pair_list = [sitk.GetArrayFromImage(sitk_pair) for sitk_pair in sitk_pair_list]
        sample = {'image': np.asarray([pair_list[0]*2.-1.,pair_list[1]*2.-1.])}

        if len(pair_path)==4:
            try:
                sample ['label']= np.asarray([pair_list[2], pair_list[3]]).astype(np.float32)
            except:
                print(pair_list[2].shape,pair_list[3].shape)
                print(filename)
        else:
            sample['label'] = None
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if sample['label'] is not None:
                 sample['label'] = self.transform(sample['label'])
        #sample['spacing'] = self.transform(sample['info']['spacing'])
        return sample,filename




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        if n_tensor.shape[0] != 1:
            n_tensor.unsqueeze_(0)
        return n_tensor
