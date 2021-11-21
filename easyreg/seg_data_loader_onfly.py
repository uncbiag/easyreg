from __future__ import print_function, division
import blosc
import torch
from torch.utils.data import Dataset
from data_pre.seg_data_utils import *
from data_pre.transform import Transform
import SimpleITK as sitk
from multiprocessing import *
blosc.set_nthreads(1)
import progressbar as pb
from copy import deepcopy
import random
import time
class SegmentationDataset(Dataset):
    """segmentation dataset.
    if the data are loaded into memory, we provide data processing option like image resampling and label filtering
    if not, for efficiency, we assume the data are preprocessed and the image resampling still works but the label filtering are disabled
    """

    def __init__(self, data_path,phase, transform=None, option = None):
        """
        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num_for_loading=option['max_num_for_loading',(-1,-1,-1,-1),"the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]"]
        self.max_num_for_loading = max_num_for_loading[ind]
        self.has_label = False
        self.get_file_list()
        self.seg_option = option['seg']
        self.img_after_resize = option[('img_after_resize', [-1, -1, -1], "numpy coordinate, resample the image into desired size")]
        self.normalize_via_percentage_clip = option[('normalize_via_percentage_clip',-1,"normalize the image via percentage clip, the given value is in [0-1]")]
        self.normalize_via_range_clip = option[('normalize_via_range_clip',(-1,-1),"normalize the image via range clip")]
        self.img_after_resize = None if any([sz == -1 for sz in self.img_after_resize]) else self.img_after_resize
        self.patch_size =  self.seg_option['patch_size']
        self.interested_label_list = self.seg_option['interested_label_list',[-1],"the label to be evaluated, the label not in list will be turned into 0 (background)"]
        self.interested_label_list = None if any([label == -1 for label in self.interested_label_list]) else self.interested_label_list
        self.transform_name_seq = self.seg_option['transform']['transform_seq']
        self.option_p = self.seg_option[('partition', {}, "settings for the partition")]
        self.use_whole_img_as_input = self.seg_option[('use_whole_img_as_input',False,"use whole image as the input")]
        self.load_into_memory = True
        self.img_list = []
        self.img_sz_list = []
        self.original_spacing_list = []
        self.original_sz_list = []
        self.spacing_list = []
        self.label_org_index_list = []
        self.label_converted_index_list = []
        self.label_density_list = []
        if self.load_into_memory:
            self.init_img_pool()
            print('img pool initialized complete')
            if self.phase=='train':
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
        if not os.path.exists(self.data_path):
            self.path_list = []
            self.name_list = []
            self.init_weight_list = []
            return
        self.path_list = read_txt_into_list(os.path.join(self.data_path, 'file_path_list.txt'))
        if len(self.path_list[0]) == 2:
            self.has_label = True
        elif self.phase in ["train", "val", "debug"]:
            raise ValueError("the label must be provided during training")
        if not self.has_label:
            self.path_list= [[path] for path in self.path_list]
        file_name_path = os.path.join(self.data_path, 'file_name_list.txt')
        if os.path.isfile(file_name_path):
            self.name_list = read_txt_into_list(file_name_path)
        else:
            self.name_list = [get_file_name(self.path_list[i][0]) for i in range(len(self.path_list))]

        if self.max_num_for_loading>0:
            read_num = min(self.max_num_for_loading, len(self.path_list))
            if self.phase=='train':
                index =list(range(len(self.path_list)))
                random.shuffle(index)
                self.path_list = [self.path_list[ind] for ind in index ]
                self.name_list = [self.name_list[ind] for ind in index ]
            self.path_list = self.path_list[:read_num]
            self.name_list = self.name_list[:read_num]


        # if len(self.name_list)==0:
        #     self.name_list = ['img_{}'.format(idx) for idx in range(len(self.path_list))]
        self.num_img = len(self.name_list)


    def __read_img_label_into_zipnp(self,img_label_path_dic,img_label_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(img_label_path_dic)).start()
        count = 0
        for fn, img_label_path in img_label_path_dic.items():
            img_label_np_dic = {}
            img_sitk, original_spacing, original_sz = self.__read_and_clean_itk_info(img_label_path['image'])
            resized_img, resize_factor = self.resize_img(img_sitk)
            img_np = sitk.GetArrayFromImage(resized_img)
            img_np = self.normalize_intensity(img_np)
            img_label_np_dic['image'] = blosc.pack_array(img_np.astype(np.float32))

            if self.has_label:
                label_sitk, _, _ = self.__read_and_clean_itk_info(img_label_path['label'])
                resized_label,_ = self.resize_img(label_sitk,is_label=True)
                label_np = sitk.GetArrayFromImage(resized_label)
                label_index = list(np.unique(label_np))
                img_label_np_dic['label'] = blosc.pack_array(label_np.astype(np.int64))
                img_label_np_dic['label_index'] = label_index
            img_after_resize = self.img_after_resize if self.img_after_resize is not None else original_sz
            new_spacing=  original_spacing*(original_sz-1)/(np.array(img_after_resize)-1)
            normalized_spacing = self._normalize_spacing(new_spacing,img_after_resize, silent_mode=True)
            img_label_np_dic['original_sz'] =original_sz
            img_label_np_dic['original_spacing'] = original_spacing
            img_label_np_dic['spacing'] = normalized_spacing
            img_label_np_dic['img_sz'] = list(img_np.shape)
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


    def __convert_to_standard_label_map(self, label_map, interested_label_list):
        label_map =blosc.unpack_array(label_map)

        cur_label_list = list(np.unique(label_map)) # unique func orders the elements
        if set(cur_label_list) == set(interested_label_list):
            return label_map

        for l_id in cur_label_list:
            if l_id in interested_label_list:
                st_index = interested_label_list.index(l_id)
            else:
                # assume background label is 0
                st_index = 0
                print("warning label: {} is not in interested label index, and would be convert to 0".format(l_id))
            label_map[np.where(label_map == l_id)] = st_index
        return label_map
    def __get_clean_label(self,img_label_dict, img_name_list):
        """

        :param img_label_dict:
        :param img_name_list:
        :return:
        """
        print(" Attention, the annotation for background is assume to be 0 ! ")
        print(" Attention, we are using the union set of the label! ")
        if self.interested_label_list is None:
            interested_label_set = set()
            for i, fname in enumerate(img_name_list):
                label_set = img_label_dict[fname]['label_index']
                if i ==0:
                    interested_label_set = set(label_set)
                else:
                    interested_label_set = interested_label_set.union(label_set)
            interested_label_list = list(interested_label_set)
        else:
            interested_label_list = self.interested_label_list

        #self.standard_label_index = tuple([int(item) for item in interested_label_list])
        for fname in img_name_list:
            label = img_label_dict[fname]['label']
            label = self.__convert_to_standard_label_map(label, interested_label_list)
            label_density = list(np.bincount(label.reshape(-1).astype(np.int32)) / len(label.reshape(-1)))
            img_label_dict[fname]['label'] = blosc.pack_array(label)
            img_label_dict[fname]['label_density']=label_density
            img_label_dict[fname]['label_org_index'] = interested_label_list
            img_label_dict[fname]['label_converted_index'] = list(range(len(interested_label_list)))
        return img_label_dict

    def init_img_pool(self):
        """img pool shoudl include following thing:
        img_label_path_dic:{img_name:{'image':img_fp,'label':label_fp,...}
        img_label_dic: {img_name:{'image':img_np,'label':label_np},......}
        img_list [[s_np,t_np,sl_np,tl_np],....]
        only the img_list need to be used by get_item method
        """
        use_parallel = self.phase=='train'
        if use_parallel:
            manager = Manager()
            img_label_dic = manager.dict()
            img_label_path_dic = {}
            img_name_list = []
            for i,fps in enumerate(self.path_list):
                fn = self.name_list[i]
                if fn not in img_label_path_dic:
                    if self.has_label:
                        img_label_path_dic[fn] = {'image':fps[0], 'label':fps[1]}
                    else:
                        img_label_path_dic[fn] = {'image':fps[0]}
                img_name_list.append(fn)
            num_of_workers = 4
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
            img_label_dic=dict(img_label_dic) # todo uncomment manager.dict
        else:
            img_label_dic=dict()
            img_label_path_dic = {}
            img_name_list = []
            for i,fps in enumerate(self.path_list):
                fn = self.name_list[i]
                if fn not in img_label_path_dic:
                    if self.has_label:
                        img_label_path_dic[fn] = {'image': fps[0], 'label': fps[1]}
                    else:
                        img_label_path_dic[fn] = {'image': fps[0]}
                img_name_list.append(fn)
            self.__read_img_label_into_zipnp(img_label_path_dic, img_label_dic) #todo dels

        self.get_organize_structure(img_label_dic,img_name_list)



    def get_organize_structure(self, img_label_dic, img_name_list):
        if self.has_label:
            img_label_dic = self.__get_clean_label(img_label_dic, img_name_list)

        for fname in img_name_list:
            if self.has_label:
                self.img_list.append([img_label_dic[fname]['image'],
                                   img_label_dic[fname]['label']])
            else:
                self.img_list.append([img_label_dic[fname]['image']])
            self.img_sz_list.append(img_label_dic[fname]['img_sz'])
            self.original_spacing_list.append(img_label_dic[fname]['original_spacing'])
            self.original_sz_list.append(img_label_dic[fname]['original_sz'])
            self.spacing_list.append(img_label_dic[fname]['spacing'])
            if self.has_label:
                self.label_org_index_list.append(img_label_dic[fname]['label_org_index'])
                self.label_converted_index_list.append(img_label_dic[fname]['label_converted_index'])
                self.label_density_list.append(img_label_dic[fname]['label_density'])

        # self.img_list = np.array(self.img_list)
        # self.img_sz_list = np.array(self.img_sz_list)
        # self.original_spacing_list = np.array(self.original_spacing_list)
        # self.original_sz_list = np.array(self.original_sz_list)
        # self.spacing_list = np.array(self.spacing_list)
        # self.label_org_index_list = np.array(self.label_org_index_list)
        # self.label_converted_index_list = np.array(self.label_converted_index_list)
        # self.label_density_list = np.array(self.label_density_list)

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
        resize_factor = np.array(img_after_resize) / np.flipud(img_sz)
        spacing_factor = (np.array(img_after_resize)-1) / (np.flipud(img_sz)-1)
        resize = not all([factor == 1 for factor in resize_factor])
        if resize:
            resampler = sitk.ResampleImageFilter()
            dimension = 3
            factor = np.flipud(resize_factor)
            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
            after_size = [int(sz) for sz in after_size]
            matrix[0, 0] = 1. / spacing_factor[0]
            matrix[1, 1] = 1. / spacing_factor[1]
            matrix[2, 2] = 1. / spacing_factor[2]
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

    def normalize_intensity(self, img):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param percen_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :param range_clip:  Linearly normalized image intensities from (range_clip[0], range_clip[1]) to 0,1
        :return
        """
        if self.normalize_via_percentage_clip>0:
            img = img - img.min()
            normalized_img = img / np.percentile(img, 95) * 0.95
        else:
            range_clip = self.normalize_via_range_clip
            if range_clip[0]<range_clip[1]:
                img = np.clip(img,a_min=range_clip[0], a_max=range_clip[1])
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img - img.min()) / (max_intensity - min_intensity)
        normalized_img = normalized_img * 2 - 1
        return normalized_img

    def __read_and_clean_itk_info(self, path):
        if path is not None:
            img = sitk.ReadImage(path)
            spacing_sitk = img.GetSpacing()
            img_sz_sitk = img.GetSize()
            return sitk.GetImageFromArray(sitk.GetArrayFromImage(img)), np.flipud(spacing_sitk), np.flipud(img_sz_sitk)
        else:
            return None, None, None

    def __read_itk_into_np(self, path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def __split_dict(self, dict_to_split, split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        split_dict = []
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj = dict(dict_to_split_items[index_split[i][0]:index_split[i][-1] + 1])
            split_dict.append(dj)
        return split_dict

    def __convert_np_to_itk_coord(self,coord_list):
        return list(np.flipud(np.array(coord_list)))



    def get_transform_seq(self,i):
        option_trans = deepcopy(self.seg_option['transform'])
        option_trans['shared_info']['label_list'] = self.label_converted_index_list[i]
        option_trans['shared_info']['label_density'] = self.label_density_list[i]
        option_trans['shared_info']['img_size'] = self.__convert_np_to_itk_coord(self.img_sz_list[i])
        option_trans['shared_info']['num_crop_per_class_per_train_img'] = self.seg_option['num_crop_per_class_per_train_img']
        option_trans['my_bal_rand_crop']['scale_ratio'] = self.seg_option['transform']['my_bal_rand_crop']['scale_ratio']
        option_trans['patch_size'] = self.__convert_np_to_itk_coord(self.seg_option['patch_size'])
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
        patch_sz_itk =self.__convert_np_to_itk_coord(self.seg_option['patch_size'])
        overlap_sz_itk =self.__convert_np_to_itk_coord(self.option_p['overlap_size'])
        self.corr_partition_pool = [deepcopy(partition(self.option_p,patch_sz_itk,overlap_sz_itk)) for _ in range(self.num_img)]



    def __len__(self):
        if self.phase == "train":
            if not self.use_whole_img_as_input:
                return len(self.name_list)*1000
            else:
                return len(self.name_list)
        else:
            return len(self.name_list)


    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        random_state = np.random.RandomState(int(time.time()))
        rand_label_id =random_state.randint(0,1000)+idx
        idx = idx%self.num_img

        filename = self.name_list[idx]
        zipnp_list = self.img_list[idx]
        spacing = self.spacing_list[idx]
        original_spacing = self.original_spacing_list[idx]
        original_sz = self.original_sz_list[idx]
        if self.has_label:
            img_np, label_np = [blosc.unpack_array(item) for item in zipnp_list]
        else:
            img_np = blosc.unpack_array(zipnp_list[0])
        img_path = self.path_list[idx]
        img_shape = img_np.shape



        if self.phase=="train":
            sample = {'image': [img_np],  'label': label_np} # here the list is for multi-modality , each mode is an elem in list
            sample = self.apply_transform(sample,self.corr_transform_pool[idx],rand_label_id)

        else:
            if not self.has_label:
                sample = {'image':  [img_np]}
            else:
                sample = {'image':  [img_np], 'label':label_np}
            if not self.use_whole_img_as_input:
                sample = self.corr_partition_pool[idx](sample)
            else:
                sample['image'] = np.stack(sample['image'], 0)
                sample['image'] = np.stack(sample['image'], 0)

        sample['img_path'] = img_path
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if self.has_label:
                sample['label'] = self.transform(sample['label'])

        sample['spacing'] = spacing.copy()
        sample["image_after_resize"] =np.array(img_shape)
        sample['original_sz'] = original_sz.copy()
        sample['original_spacing'] = original_spacing.copy()
        return sample, filename





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor
