from __future__ import print_function, division
import blosc
import torch
from torch.utils.data import Dataset
from .reg_data_utils import *
import SimpleITK as sitk
from multiprocessing import *
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
        : seg_option: pars,  settings for segmentation task,  None for segmentation task
        : reg_option:  pars, settings for registration task, None for registration task

        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        #self.data_type = '*.nii.gz'
        self.turn_on_pair_regis = False
        """ inverse the registration order, i.e the original set is A->B, the new set would be A->B and B->A """
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num_for_loading=reg_option['max_num_for_loading',(-1,-1,-1,-1),"the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]"]
        self.max_num_for_loading = max_num_for_loading[ind]
        """ the max number of pairs to be loaded into the memory,[max_train, max_val, max_test, max_debug]"""
        self.load_init_weight=reg_option[('load_init_weight',False,'load init weight for adaptive weighting model')]
        self.has_label = False
        self.get_file_list()
        self.reg_option = reg_option
        self.img_after_resize = reg_option[('img_after_resize',[-1,-1,-1],"resample the image into desired size")]
        self.img_after_resize = None if any([sz == -1 for sz in self.img_after_resize]) else self.img_after_resize
        load_training_data_into_memory = reg_option[('load_training_data_into_memory',False,"when train network, load all training sample into memory can relieve disk burden")]
        self.load_into_memory = load_training_data_into_memory if phase == 'train' else False
        self.pair_list = []
        self.original_spacing_list = []
        self.original_sz_list = []
        self.spacing_list = []
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


        if self.max_num_for_loading>0:
            read_num = min(self.max_num_for_loading, len(self.path_list))
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
            img_sitk, original_spacing, original_sz = self.__read_and_clean_itk_info(img_label_path['img'])
            resized_img, resize_factor = self.resize_img(img_sitk)
            img_np = sitk.GetArrayFromImage(resized_img)
            img_label_np_dic['img'] = blosc.pack_array(img_np.astype(np.float32))

            if self.has_label:
                label_sitk, _, _ = self.__read_and_clean_itk_info(img_label_path['label'])
                resized_label,_ = self.resize_img(label_sitk,is_label=True)
                label_np = sitk.GetArrayFromImage(resized_label)
                img_label_np_dic['label'] = blosc.pack_array(label_np.astype(np.float32))
            new_spacing=  original_spacing*(original_sz-1)/(np.array(self.img_after_resize)-1)
            normalized_spacing = self._normalize_spacing(new_spacing,self.img_after_resize, silent_mode=True)
            img_label_np_dic['original_sz'] =original_sz
            img_label_np_dic['original_spacing'] = original_spacing
            img_label_np_dic['spacing'] = normalized_spacing
            img_label_dic[fn] =img_label_np_dic
            count +=1
            pbar.update(count)
        pbar.finish()



    def _normalize_spacing(self,spacing,sz,silent_mode=False):
        """
        Normalizes spacing.
        :param spacing: Vector with spacing info, in XxYxZ format
        :param sz: size vector in XxYxZ format
        :return: vector with normalized spacings in XxYxZ format
        """
        dim = len(spacing)
        # first determine the largest extent
        current_largest_extent = -1
        extent = np.zeros_like(spacing)
        for d in range(dim):
            current_extent = spacing[d]*(sz[d]-1)
            extent[d] = current_extent
            if current_extent>current_largest_extent:
                current_largest_extent = current_extent

        scalingFactor = 1./current_largest_extent
        normalized_spacing = spacing*scalingFactor

        normalized_extent = extent*scalingFactor

        if not silent_mode:
            print('Normalize spacing: ' + str(spacing) + ' -> ' + str(normalized_spacing))
            print('Normalize spacing, extent: ' + str(extent) + ' -> ' + str(normalized_extent))

        return normalized_spacing




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
        num_of_workers = 12
        num_of_workers = num_of_workers if len(self.name_list)>12 else 2
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

            self.original_spacing_list.append(img_label_dic[sn]['original_spacing'])
            self.original_sz_list.append(img_label_dic[sn]['original_sz'])
            self.spacing_list.append(img_label_dic[sn]['spacing'])





    def resize_img(self, img, is_label=False):
        """
        :param img: sitk input, factor is the outputs_ize/patched_sized
        :return:
        """
        img_sz = img.GetSize()
        if self.img_after_resize is not None:
            img_after_resize = self.img_after_resize
        else:
            img_after_resize = np.flipud(img_sz)
        resize_factor = np.array(img_after_resize)/np.flipud(img_sz)
        resize = not all([factor == 1 for factor in resize_factor])
        if resize:
            resampler= sitk.ResampleImageFilter()
            dimension =3
            factor = np.flipud(resize_factor)
            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [round(img_sz[i]*factor[i]) for i in range(dimension)]
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
        else:
            img_resampled = img
        return img_resampled, resize_factor

    def normalize_intensity(self, img, linear_clip=False):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """
        if linear_clip:
            img = img - img.min()
            normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        normalized_img = normalized_img*2 - 1
        return normalized_img


    def __read_and_clean_itk_info(self,path):
        if path is not None:
            img = sitk.ReadImage(path)
            spacing_sitk = img.GetSpacing()
            img_sz_sitk = img.GetSize()
            return sitk.GetImageFromArray(sitk.GetArrayFromImage(img)), np.flipud(spacing_sitk), np.flipud(img_sz_sitk)
        else:
            return None, None, None

    def __read_itk_into_np(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def __split_dict(self,dict_to_split,split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list),split_num)
        split_dict=[]
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj=dict(dict_to_split_items[index_split[i][0]:index_split[i][-1]+1])
            split_dict.append(dj)
        return split_dict

    def __inverse_name(self,name):
        """get the name of the inversed registration pair"""
        name = name+'_inverse'
        return name
        # try:
        #     n_parts= name.split('_image_')
        #     inverse_name = n_parts[1]+'_'+n_parts[0]+'_image'
        #     return inverse_name
        # except:
        #     n_parts = name.split('_brain_')
        #     inverse_name = n_parts[1] + '_' + n_parts[0] + '_brain'
        #     return inverse_name



    def __len__(self):
        return len(self.name_list)*500 if len(self.name_list)<100 and self.phase=='train' else len(self.name_list)  #############################3



    def __getitem__(self, idx):
        """
        # todo  update the load data part to mermaid fileio
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        # print(idx)
        idx = idx %len(self.name_list)

        pair_path = self.path_list[idx]
        filename = self.name_list[idx]
        if not self.load_into_memory:
            img_spacing_pair_list = [ list(self.__read_and_clean_itk_info(pt)) for pt in pair_path]
            sitk_pair_list = [item[0] for item in img_spacing_pair_list]
            original_spacing = img_spacing_pair_list[0][1]
            original_sz = img_spacing_pair_list[0][2]
            sitk_pair_list[0], resize_factor = self.resize_img(sitk_pair_list[0])
            sitk_pair_list[1], _ = self.resize_img(sitk_pair_list[1])
            if self.has_label:
                sitk_pair_list[2],_ = self.resize_img(sitk_pair_list[2], is_label=True)
                sitk_pair_list[3],_ = self.resize_img(sitk_pair_list[3], is_label=True)
            pair_list = [sitk.GetArrayFromImage(sitk_pair) for sitk_pair in sitk_pair_list]
            img_after_resize = self.img_after_resize if self.img_after_resize is not None else original_sz
            new_spacing=  original_spacing*(original_sz-1)/(np.array(img_after_resize)-1)
            spacing = self._normalize_spacing(new_spacing,img_after_resize, silent_mode=True)


        else:
            zipnp_pair_list = self.pair_list[idx]
            spacing = self.spacing_list[idx]
            original_spacing = self.original_spacing_list[idx]
            original_sz = self.original_sz_list[idx]
            pair_list = [blosc.unpack_array(item) for item in zipnp_pair_list]

        sample = {'image': np.asarray([self.normalize_intensity(pair_list[0]),self.normalize_intensity(pair_list[1])])}
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

        sample['spacing'] = spacing.copy()
        sample['original_sz'] = original_sz.copy()
        sample['original_spacing'] = original_spacing.copy()
        return sample,filename




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor


