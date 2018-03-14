import torch
from queue import PriorityQueue
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_pre.seg_data_loader import SegmentationDataset
from data_pre.seg_data_utils import *
from queue import PriorityQueue
import re
from torch._six import string_classes
import collections
import random


class SegmentationDatasetPQ(Dataset):
    """registration dataset."""

    def __init__(self, data_path,is_train=True,num_pq = 10):
        """

        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.is_train = is_train
        self.transform = ToTensor()
        self.data_type = '*.h5py'
        self.path_list , self.name_list= self.get_file_list()
        self.name_path_dic = {name: self.path_list[i] for i, name in enumerate(self.name_list) }
        self.num_pq = num_pq
        self.data_pq_record = [PriorityQueue()]*self.num_pq
        self.init_priority_queue()

    def init_priority_queue(self):
        rand_id = np.random.permutation(len(self.name_list))
        # to avoid bug in pq, so here add rand id to distinguish
        rand_id_par = np.array_split(rand_id, self.num_pq)

        index_id_par = np.array_split(range(len(self.name_list),self.num_pq))
        for id_pq in range(self.num_pq):
            for id in index_id_par[id_pq]:
                self.data_pq_record[id_pq].put((0.,rand_id_par[id_pq][id],{'name':self.name_list[id],
                                            'count':0,
                                            'l_score':[0.]*3,
                                            'path':self.name_path_dic[self.name_list[id]],
                                            'l_den':[0]*3,
                                            'l_list':[0]*3,
                                            'pq_id':id_pq})
                                    )
    def put_into_pq(self,patch_item):
        """

        :param patch_item: is a tuple (score,dic)
        :return:
        """
        self.data_pq_record.put(patch_item)

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        f_filter = glob( join(self.data_path, '**', '*.h5py'), recursive=True)
        name_list = [get_file_name(f,last_ocur=True) for f in f_filter]
        index = list(range(len(f_filter)))
        random.shuffle(index)
        f_filter = [f_filter[i] for i in index]
        name_list = [name_list[i] for i in index]
        return f_filter,name_list

    def get_cur_pq_len(self):
        return self.data_pq_record.qsize()

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, idx):
        """
        :param idx: is no use here
        :return: the processed data, return as type of dic
        """
        patch_item = self.data_pq_record.get()
        path = patch_item[2]['path']
        fname  = patch_item[2]['name']
        dic = read_h5py_file(path)
        sample = {'image': dic['data'], 'info': dic['info'], 'label':dic['label']}
        if self.transform:
            sample['image'] = self.transform(sample['image'].astype(np.float32))
            if sample['label'] is not None:
                 sample['label'] = self.transform(sample['label'].astype(np.int32))
        return sample,fname,patch_item



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





numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch,_use_shared_memory=False):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))



class DataLoaderPQ(object):
    def __init__(self,dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def get_batch(self):
        samples = default_collate([self.dataset[0] for _ in range(self.batch_size)])
        return samples

    def insert_pq(self,item):
        self.dataset.put_into_pq(item)

    def get_cur_pq_len(self):
        return self.dataset.get_cur_pq_len()

    def __iter__(self):
        return self

    def __next__(self):
        if self.get_cur_pq_len()>self.batch_size:
            return self.get_batch()
        else:
            raise StopIteration


if __name__ == "__main__":
    data_path = '/playpen/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/for_debug'
    batch_size = 3
    dataset = SegmentationDatasetPQ(data_path)
    data_loader = DataLoaderPQ(dataset, batch_size)
    count = 48
    i = 0
    for data in data_loader:
        batch_item = data[2]
        for j in range(batch_size):
            print('the {} iter,the batch name{}, score{}, batch_id{},count{} '.format(i,batch_item[2]['name'][j], batch_item[0][j],batch_item[1][j],batch_item[2]['count'][j]))
            score = (batch_item[0][j]+1.)/10
            info = batch_item[2]
            batch_item[2]['count'][j]+= 1
            updated_patch =(score,batch_item[1][j],{name:value[j] for name,value in info.items()})
            data_loader.insert_pq(updated_patch)
            if i> count:
                break
            i +=1
