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
from copy import deepcopy
from data_pre.transform import Transform
import blosc


class SegmentationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path,phase, transform=None, option = None):
        """
        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.is_train = phase =='train'
        self.is_test = phase=='test'
        self.phase = phase
        self.transform = transform
        self.data_type = '*.h5py'
        self.path_list , self.name_list= self.get_file_list()
        self.num_img = len(self.path_list)
        self.transform_name_seq = option['transform']['transform_seq']
        self.option_p = option[('partition', {}, "settings for the partition")]
        self.option_p['patch_size'] = option['patch_size']
        self.option = option
        self.img_pool = []
        if self.is_train:
            self.init_img_pool()
            print('img pool initialized complete')
            self.init_corr_transform_pool()
            print('transforms initialized complete')
        else:
            self.init_corr_partition_pool()
            print("partition pool initialized complete")
        blosc.set_nthreads(1)

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        f_filter = glob( join(self.data_path, '**', '*.h5py'), recursive=True)
        ##############

        # f_filter= f_filter[0:3]
        #############
        name_list = [get_file_name(f,last_ocur=True) for f in f_filter]
        assert len(f_filter)==len(name_list)
        return f_filter,name_list


    def get_transform_seq(self,i):
        option_trans = deepcopy(self.option)
        option_trans['shared_info']['label_list'] = self.img_pool[i]['info']['label_list']
        option_trans['shared_info']['label_density'] = self.img_pool[i]['info']['label_density']
        option_trans['shared_info']['img_size'] = self.img_size
        option_trans['shared_info']['num_crop_per_class_per_train_img'] = self.option['num_crop_per_class_per_train_img']
        if len(self.img_pool[i]['info']['label_list'])==3:
            print(self.name_list[i])
        transform = Transform(option_trans)
        return transform.get_transform_seq(self.transform_name_seq)


    def apply_transform(self,sample, transform_seq):
        for transform in transform_seq:
            sample = transform(sample)
        return sample



    def init_corr_transform_pool(self):
        self.corr_transform_pool = [self.get_transform_seq(i) for i in range(self.num_img)]
    def init_corr_partition_pool(self):
        from data_pre.partition import partition
        if not self.is_test:
            self.corr_partition_pool = [deepcopy(partition(self.option_p)) for i in range(self.num_img)]
        else:
            self.corr_partition_pool = [deepcopy(partition(self.option_p, mode='pred')) for i in range(self.num_img)]

    def init_img_pool(self):
        for path in self.path_list:
            dic = read_h5py_file(path)
            sample = { 'info': dic['info']}
            folder_path = os.path.split(path)[0]
            mode_filter  = glob(join(folder_path,'*tmod*.nii.gz'),recursive=True)
            img_list_sitk=[]
            for mode_path in mode_filter:
                img_list_sitk += [sitk.ReadImage(mode_path)]
            modes = sitk_to_np(img_list_sitk)
            modes_pack = blosc.pack_array(modes)
            sample['img'] =modes_pack
            seg_path =path.replace('.h5py','_seg.nii.gz')
            sample['seg'] = sitk_to_np(sitk.ReadImage(seg_path))
            self.img_size = list(sample['seg'].shape)[1:]
            self.img_pool += [sample]

            # img_path_list = []
            # for mode_path in mode_filter:
            #     img_path_list += [mode_path]
            # sample['image_path_list'] = img_path_list
            # seg_path = path.replace('.h5py', '_seg.nii.gz')
            # sample['label_path'] = seg_path
            # self.img_pool += [sample]

    def __len__(self):
        if self.is_train:
            return len(self.name_list)*100
        else:
            return len(self.name_list)


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        idx = idx%self.num_img
        fname  = self.name_list[idx]+'_tile'
        if self.is_train:
            dic = self.img_pool[idx]
            modes = blosc.unpack_array(dic['img'])
            data = {'img': modes, 'info': dic['info'], 'seg': dic['seg']}

            data['img'] = [modes[i] for i in range(modes.shape[0])]
            data = self.apply_transform(data,self.corr_transform_pool[idx])
        else:
            dic = read_h5py_file(self.path_list[idx])
            data = {'img': dic['data'], 'info': dic['info'], 'seg': dic['label']}
        sample = {}

        ########################### channel sel
        if self.transform:
            index = [0,1,2,3]
            if self.is_train:
                sample['image'] = self.transform(data['img'][index].copy())*2-1
            else:
                sample['image'] = self.transform(data['img'][:,index].copy())*2-1
            #sample['image'] = self.transform(data['img'].copy())
            if data['seg'] is not None:
                 sample['label'] = self.transform(data['seg'].copy())
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