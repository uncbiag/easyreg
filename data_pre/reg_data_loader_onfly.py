from __future__ import print_function, division
import os

import blosc
import torch
from torch.utils.data import Dataset, DataLoader
from data_pre.reg_data_utils import *
from multiprocessing import *
num_of_workers = 12
blosc.set_nthreads(1)
import progressbar as pb

class RegistrationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path,phase=None, transform=None, seg_option=None, reg_option=None):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs are compressed into memory
        :param data_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        :param transform: function,  apply transform on data
        : seg_option: pars,  settings for segmentation task,  None for registration task
        : reg_option:  pars, settings for registration task, None for segmentation task

        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.data_type = '*.nii.gz'
        self.turn_on_pair_regis = False
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        self.max_num_pair_to_load = reg_option['max_pair_for_loading'][ind]
        """ the max number of pairs to be loaded into the memory"""
        self.load_init_weight=reg_option[('load_init_weight',False,'load init weight for adaptive weighting model')]


        # ##########################ToDO  delete this section #################################3
        # use_extra_inter_intra_judge =True
        # self.is_intra_reg = True
        # if use_extra_inter_intra_judge:
        #     self.is_intra_reg = True if 'intra' in data_path else False
        #
        # if self.is_intra_reg:
        #     self.max_num_pair_to_load = -1 if phase != 'test' else 300  # 300 when test intra    150     ###################TODO ###################################3
        #     self.turn_on_pair_regis = True if phase != 'test' else False  # True when test inter   ##########TODO ########################
        # else:
        #     self.max_num_pair_to_load = -1 if phase != 'test' else 150  # 300 when test intra    150     ###################TODO ###################################3
        #     self.turn_on_pair_regis = True  # if ph

        ######################################################################################3

        self.has_label = False
        self.get_file_list()
        self.reg_option = reg_option
        self.resize_factor = reg_option['input_resize_factor']
        self.resize = not all([factor==1 for factor in self.resize_factor])
        load_training_data_into_memory = reg_option['load_training_data_into_memory']
        self.load_into_memory = load_training_data_into_memory if phase == 'train' else False
        self.pair_list = []
        if self.load_into_memory:
            self.init_img_pool()

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        if not os.path.exists(self.data_path):
            self.path_list=[]
            self.name_list=[]
            self.init_weight_list=[]
            return
        self.path_list = read_txt_into_list(os.path.join(self.data_path,'pair_path_list.txt'))
        self.name_list = read_txt_into_list(os.path.join(self.data_path, 'pair_name_list.txt'))
        if self.load_init_weight:
            self.init_weight_list = read_txt_into_list(os.path.join(self.data_path,'pair_weight_path_list.txt'))
        if len(self.path_list[0])==4:
            self.has_label=True


        if self.max_num_pair_to_load>0:
            read_num = min(self.max_num_pair_to_load, len(self.path_list))
            self.path_list = self.path_list[:read_num]
            self.name_list = self.name_list[:read_num]
            if self.load_init_weight:
                self.init_weight_list = self.init_weight_list[:read_num]

        if self.turn_on_pair_regis and (self.phase=='train' or self.phase == 'test'): #self.phase =='test' and self.turn_on_pair_regis:
            path_list_inverse = [[path[1],path[0], path[3], path[2]] for path in self.path_list]
            name_list_inverse = [self.__inverse_name(name) for name in self.name_list]
            self.path_list += path_list_inverse
            self.name_list += name_list_inverse
            if self.load_init_weight:
                init_weight_inverse =[[path[1],path[0]] for path in self.init_weight_list]
                self.init_weight_list += init_weight_inverse

        if len(self.name_list)==0:
            self.name_list = ['pair_{}'.format(idx) for idx in range(len(self.path_list))]

    def __read_img_label_into_zipnp(self,img_label_path_dic,img_label_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(img_label_path_dic)).start()
        count = 0
        for fn, img_label_path in img_label_path_dic.items():
            img_label_np_dic = {}
            img_sitk = self.__read_and_clean_itk_info(img_label_path['img'])
            if self.has_label:
                label_sitk = self.__read_and_clean_itk_info(img_label_path['label'])
            if self.resize:
                img_np = sitk.GetArrayFromImage(self.resize_img(img_sitk))
                if self.has_label:
                    label_np = sitk.GetArrayFromImage(self.resize_img(label_sitk,is_label=True))
            else:
                img_np = sitk.GetArrayFromImage(img_sitk)
                if self.has_label:
                    label_np = sitk.GetArrayFromImage(label_sitk)

            img_label_np_dic['img'] = blosc.pack_array(img_np.astype(np.float32))
            if self.has_label:
                img_label_np_dic['label'] = blosc.pack_array(label_np.astype(np.float32))
            img_label_dic[fn] =img_label_np_dic
            count +=1
            pbar.update(count)
        pbar.finish()




    def init_img_pool(self):
        """img pool shoudl include following thing:
        img_label_path_dic:{img_name:{'img':img_fp,'label':label_fp,...}
        img_label_dic: {img_name:{'img':img_np,'label':label_np},......}
        pair_name_list:[[pair1_s,pair1_t],[pair2_s,pair2_t],....]
        pair_list [[s_np,t_np,sl_np,tl_np],....]
        only the pair_list need to be used by get_item method
        """
        manager = Manager()
        img_label_dic = manager.dict()
        img_label_path_dic = {}
        pair_name_list = []
        for fps in self.path_list:
            for i in range(2):
                fp = fps[i]
                fn = get_file_name(fp)
                if fn not in img_label_path_dic:
                    if self.has_label:
                        img_label_path_dic[fn] = {'img':fps[i], 'label':fps[i+2]}
                    else:
                        img_label_path_dic[fn] = {'img':fps[i]}
            pair_name_list.append([get_file_name(fps[0]), get_file_name(fps[1])])



        split_dict = self.__split_dict(img_label_path_dic,num_of_workers)
        procs =[]
        for i in range(num_of_workers):
            p = Process(target=self.__read_img_label_into_zipnp,args=(split_dict[i], img_label_dic,))
            p.start()
            print("pid:{} start:".format(p.pid))

            procs.append(p)

        for p in procs:
            p.join()

        print("the loading phase finished, total {} img and labels have been loaded".format(len(img_label_dic)))
        img_label_dic=dict(img_label_dic)

        for pair_name in pair_name_list:
            sn = pair_name[0]
            tn = pair_name[1]
            if self.has_label:
                self.pair_list.append([img_label_dic[sn]['img'],img_label_dic[tn]['img'],
                                   img_label_dic[sn]['label'],img_label_dic[tn]['label']])
            else:
                self.pair_list.append([img_label_dic[sn]['img'], img_label_dic[tn]['img']])




    def resize_img(self, img, is_label=False):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud(self.resize_factor)
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
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
        if path is not None:
            return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))
        else:
            return None

    def __read_itk_into_np(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def __split_dict(self,dict_to_split,split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list),num_of_workers)
        split_dict=[]
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj=dict(dict_to_split_items[index_split[i][0]:index_split[i][-1]+1])
            split_dict.append(dj)
        return split_dict

    def __inverse_name(self,name):
        try:
            n_parts= name.split('_image_')
            inverse_name = n_parts[1]+'_'+n_parts[0]+'_image'
            return inverse_name
        except:
            n_parts = name.split('_brain_')
            inverse_name = n_parts[1] + '_' + n_parts[0] + '_brain'
            return inverse_name



    def __len__(self):
        return len(self.name_list) #############################3


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        #idx=0

        pair_path = self.path_list[idx]
        filename = self.name_list[idx]
        if not self.load_into_memory:
            sitk_pair_list = [ self.__read_and_clean_itk_info(pt) for pt in pair_path]
            if self.resize:
                sitk_pair_list[0] = self.resize_img(sitk_pair_list[0])
                sitk_pair_list[1] = self.resize_img(sitk_pair_list[1])
                if self.has_label:
                    sitk_pair_list[2] = self.resize_img(sitk_pair_list[2], is_label=True)
                    sitk_pair_list[3] = self.resize_img(sitk_pair_list[3], is_label=True)
            pair_list = [sitk.GetArrayFromImage(sitk_pair) for sitk_pair in sitk_pair_list]

        else:
            zipnp_pair_list = self.pair_list[idx]
            pair_list = [blosc.unpack_array(item) for item in zipnp_pair_list]

        sample = {'image': np.asarray([pair_list[0]*2.-1.,pair_list[1]*2.-1.])}
        sample['pair_path'] = pair_path
        if self.load_init_weight:
            sample['init_weight']=self.init_weight_list[idx]

        if self.has_label:
            try:
                sample ['label']= np.asarray([pair_list[2], pair_list[3]]).astype(np.float32)
            except:
                print(pair_list[2].shape,pair_list[3].shape)
                print(filename)
        # else:
        #     sample['label'] = None
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if self.has_label:
                 sample['label'] = self.transform(sample['label'])
        #sample['spacing'] = self.transform(sample['info']['spacing'])
        return sample,filename




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor
