from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_pre.seg_data_utils import *
from time import time


class SegmentationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path,is_train=True, transform=None):
        """

        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.is_train = is_train
        self.transform = transform
        self.data_type = '*.h5py'
        self.path_list , self.name_list= self.get_file_list()

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        f_filter = glob( join(self.data_path, '**', '*.h5py'), recursive=True)
        name_list = [get_file_name(f,last_ocur=True) for f in f_filter]
        return f_filter,name_list


    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        dic = read_h5py_file(self.path_list[idx])
        fname  = self.name_list[idx]
        sample = {'image': dic['data'], 'info': dic['info'], 'label':dic['label']}
        if self.transform:
            sample['image'] = self.transform(sample['image'].astype(np.float32))
            if sample['label'] is not None:
                 sample['label'] = self.transform(sample['label'].astype(np.int32))
        return sample,fname



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, old_version=False):

        # older version
        if old_version:
            n_tensor= torch.from_numpy(sample)
            if n_tensor.shape[0]!=1:
                n_tensor.unsqueeze_(0)
            return n_tensor
        else:
            n_tensor = torch.from_numpy(sample)
            return n_tensor
