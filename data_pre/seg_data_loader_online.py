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

# count = [1, 2, 3]


class SegmentationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path,phase, transform=None, seg_option = None,reg_option=None):
        """
        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        super(SegmentationDataset).__init__()
        self.data_path = data_path
        self.is_train = phase =='train'
        self.is_brats = 'brats' in data_path
        self.is_test = phase=='test' if self.is_brats else False
        self.phase = phase
        self.transform = transform
        self.data_type = '*.h5py'
        self.path_list , self.name_list= self.get_file_list()
        self.num_img = len(self.path_list)
        self.patch_size =  seg_option['patch_size']
        self.transform_name_seq = seg_option['transform']['transform_seq']
        self.option_p = seg_option[('partition', {}, "settings for the partition")]
        self.add_resampled = seg_option['add_resampled']
        self.add_loc = seg_option['add_loc']
        self.use_org_size = seg_option['use_org_size']
        self.detect_et = seg_option['detect_et']
        self.option_p['patch_size'] = seg_option['patch_size']
        self.seg_option = seg_option
        self.img_pool = []
        if self.is_train:
            self.init_img_pool()
            print('img pool initialized complete')
            self.init_corr_transform_pool()
            print('transforms initialized complete')
        else:
            self.init_img_pool()
            print('img pool initialized complete')
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

        #f_filter= f_filter[0:4]
        #############
        name_list = [get_file_name(f,last_ocur=True) for f in f_filter]
        return f_filter,name_list


    def get_transform_seq(self,i):
        option_trans = deepcopy(self.seg_option['transform'])
        option_trans['shared_info']['label_list'] = self.img_pool[i]['info']['label_list']
        option_trans['shared_info']['label_density'] = self.img_pool[i]['info']['label_density']
        option_trans['shared_info']['img_size'] = self.img_size
        option_trans['shared_info']['num_crop_per_class_per_train_img'] = self.seg_option['num_crop_per_class_per_train_img']
        option_trans['my_bal_rand_crop']['scale_ratio'] = self.seg_option['transform']['my_bal_rand_crop']['scale_ratio']
        option_trans['patch_size'] = self.seg_option['patch_size']

        if len(self.img_pool[i]['info']['label_list'])==3:
            print(self.name_list[i])
        transform = Transform(option_trans)
        return transform.get_transform_seq(self.transform_name_seq)


    def apply_transform(self,sample, transform_seq, rand_label_id=-1):
        for transform in transform_seq:
            sample = transform(sample, rand_label_id)
        return sample



    def init_corr_transform_pool(self):
        self.corr_transform_pool = [self.get_transform_seq(i) for i in range(self.num_img)]
    def init_corr_partition_pool(self):
        from data_pre.partition import partition
        if not self.is_test:
            self.corr_partition_pool = [deepcopy(partition(self.option_p)) for i in range(self.num_img)]
        else:
            self.corr_partition_pool = [deepcopy(partition(self.option_p, mode='pred')) for i in range(self.num_img)]

    def resize_img(self, img):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = [1,1,1]
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        matrix[0, 0] = img_sz[0]/float((self.patch_size[0]*factor[0]))
        matrix[1, 1] = img_sz[1]/float((self.patch_size[1]*factor[1]))
        matrix[2, 2] = img_sz[2]/float((self.patch_size[2]*factor[2]))
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize([dim_size*factor[i] for i,dim_size in enumerate(self.patch_size)])
        resampler.SetTransform(affine)
        img_resampled = resampler.Execute(img)
        return img_resampled


    def init_img_pool(self):
        for path in self.path_list:
            dic = read_h5py_file(path)
            sample = {'info': dic['info']}
            folder_path = os.path.split(path)[0]
            mode_filter  = glob(join(folder_path,'*tmod*.nii.gz'),recursive=True)
            img_list_sitk=[]
            img_resampled_list_sitk = []
            for mode_path in mode_filter:
                sitk_image = sitk.ReadImage(mode_path)
                img_list_sitk += [sitk_image]
                img_resampled_list_sitk += [self.resize_img(sitk_image)]
            modes = sitk_to_np(img_list_sitk)
            modes_pack = blosc.pack_array(modes)
            resampled_modes = sitk_to_np((img_resampled_list_sitk))
            resampled_modes_pack = blosc.pack_array(resampled_modes)
            sample['img'] =modes_pack
            sample['resampled_img'] = resampled_modes_pack
            if not self.is_test:
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
            # self.img_pool += [sample]`

    def __len__(self):
        if self.is_train:
            if not self.use_org_size:
                return len(self.name_list)*100
            else:
                return len(self.name_list)*10
        else:
            return len(self.name_list)
    def gen_coord_map(self, img_sz):
        map = np.mgrid[0.:1.:1./img_sz[0],0.:1.:1./img_sz[1],0.:1.:1./img_sz[2]]
        map = np.array(map.astype('float32'))

        return map
    def get_seg_info_list(self,seg):
        check_label = np.sum(seg==3)
        check_label = 1 if check_label>0 else 0
        return np.array([check_label])


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        rand_label_id =random.randint(0,1000)
        idx = idx%self.num_img
        fname  = self.name_list[idx]+'_tile'
        dic = self.img_pool[idx]
        modes = blosc.unpack_array(dic['img'])
        if self.add_loc:
            map = self.gen_coord_map(modes.shape[1:]).copy()
            modes = np.concatenate((modes, map), 0)

        if self.is_train:
            data = {'img': modes, 'info': dic['info'], 'seg': dic['seg']}
            data['img'] = [modes[i] for i in range(modes.shape[0])]
            if not self.use_org_size:
                data = self.apply_transform(data,self.corr_transform_pool[idx],rand_label_id)
            else:
                data['img'] = np.stack(data['img'],0)
        else:
            if self.is_test:
                data = {'img': modes, 'info': dic['info']}
            else:
                data = {'img': modes, 'info': dic['info'], 'seg': dic['seg']}
            data['img'] = [modes[i] for i in range(modes.shape[0])]
            if not self.use_org_size:
                data = self.corr_partition_pool[idx](data)
            else:
                data['img'] = np.stack(data['img'], 0)
        if self.detect_et and 'seg'in data and data['seg'] is not None:
            data['checked_label'] = self.get_seg_info_list(data['seg'])
        sample = {}

        ########################### channel sel
        if self.transform:
            index = [0] if not self.is_brats else [0,1,2,3]
            if self.add_loc:
                index = index + [index[-1]+1 for _ in range(3)]
            if self.is_train:
                sample['image'] = self.transform(data['img'][index].copy())*2-1
            else:
                if not self.use_org_size:
                    sample['image'] = self.transform(data['img'][:,index].copy())*2-1
                else:
                    sample['image'] = self.transform(data['img'][index].copy()) * 2 - 1
            if self.add_resampled:
                sample['resampled_img'] = self.transform(blosc.unpack_array((dic['resampled_img'])))
            if  self.detect_et: #self.use_org_size or self.detect_et:
                sample['checked_label']=self.transform(data['checked_label'].astype(np.int32))
            #sample['image'] = self.transform(data['img'].copy())
            if 'seg'in data and data['seg'] is not None:
                 sample['label'] = self.transform(data['seg'].copy())
            else:
                sample['label'] = self.transform(np.array([-1]).astype((np.int32)))
        # global count
        # count[1] +=1
        # print(count)
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