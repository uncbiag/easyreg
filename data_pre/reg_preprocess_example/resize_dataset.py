"""
this is a transfer tool for existed oai data (train, val , test, debug)
it will crop the image into desired size
include two part
crop the data and save it into new direc ( the code will be based on dataloader)
generate a new directory which should provide the same train,val,test and debug set, but point to the cropped data
"""
from __future__ import print_function, division

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../model_pool'))

from easyreg.reg_data_utils import *
from multiprocessing import *
import SimpleITK as sitk
num_of_workers = 12
import progressbar as pb
from easyreg.utils import *


class RegistrationDataset(object):
    """registration dataset."""

    def __init__(self, data_path, phase=None, resize_factor=[1., 1., 1.]):
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
        self.task_output_path = None
        self.data_output_path = None
        self.real_img_path = None
        self.real_label_path = None
        self.running_read_path = None
        self.phase = phase
        self.data_type = '*.nii.gz'
        self.turn_on_pair_regis = False
        self.max_num_pair_to_load = [-1, -1, -1, -1]
        """ the max number of pairs to be loaded into the memory"""
        self.has_label = False
        self.shared_label_set = None
        self.get_file_list()
        self.resize_factor = resize_factor
        self.resize = not all([factor == 1 for factor in self.resize_factor])
        self.pair_list = []

    def process(self):
        self.transfer_exist_dataset_txt_into_new_one()
        #self.process_img_pool()

    def set_task_output_path(self, path):
        self.task_output_path = path
        os.makedirs(path, exist_ok=True)

    def set_data_output_path(self, path):
        self.data_output_path = path
        os.makedirs(path, exist_ok=True)

    def set_real_data_path(self, img_path, label_path):
        self.real_img_path = img_path
        self.real_label_path = label_path

    def set_running_read_path(self, running_read_path):
        self.running_read_path = running_read_path

    def set_shared_label(self, shared_label_set):
        self.shared_label_set = shared_label_set

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        if not os.path.exists(self.data_path):
            self.path_list = []
            self.name_list = []
            return
        self.path_list = read_txt_into_list(os.path.join(self.data_path, 'pair_path_list.txt'))
        self.name_list = read_txt_into_list(os.path.join(self.data_path, 'pair_name_list.txt'))
        if len(self.path_list[0]) == 4:
            self.has_label = True
        if len(self.name_list) == 0:
            self.name_list = ['pair_{}'.format(idx) for idx in range(len(self.path_list))]

        if self.phase == 'test':
            self.path_list = [[pth.replace('zhenlinx/Data/OAI_segmentation', 'zyshen/oai_data') for pth in pths] for
                              pths in
                              self.path_list]

    def transfer_exist_dataset_txt_into_new_one(self):
        source_path_list, target_path_list, l_source_path_list, l_target_path_list = self.split_file_list()
        file_num = len(source_path_list)
        assert len(source_path_list) == len(target_path_list)
        if l_source_path_list is not None and l_target_path_list is not None:
            assert len(source_path_list) == len(l_source_path_list)
            file_list = [[source_path_list[i], target_path_list[i], l_source_path_list[i], l_target_path_list[i]] for i
                         in range(file_num)]
        else:
            file_list = [[source_path_list[i], target_path_list[i]] for i in range(file_num)]
        img_output_path = os.path.join(self.running_read_path, 'img')
        label_output_path = os.path.join(self.running_read_path, 'label')
        file_list = [[pths[i].replace(self.real_img_path, img_output_path) if i in [0, 1] else pths[i].replace(
            self.real_label_path[0],
            label_output_path)
                      for i, pth in enumerate(pths)] for pths in file_list]
        if len(self.real_label_path) > 1:
            file_list = [[pths[i].replace(self.real_img_path, img_output_path) if i in [0, 1] else pths[i].replace(
                self.real_label_path[1],
                label_output_path)
                          for i, pth in enumerate(pths)] for pths in file_list]
        output_path = self.task_output_path
        pair_txt_path = os.path.join(output_path, 'pair_path_list.txt')
        fn_txt_path = os.path.join(output_path, 'pair_name_list.txt')
        fname_list = [generate_pair_name([file_list[i][0],file_list[i][1]]) for i in range(file_num)]
        write_list_into_txt(pair_txt_path, file_list)
        write_list_into_txt(fn_txt_path, fname_list)

    def split_file_list(self):
        path_list = self.path_list
        num_pair = len(path_list)
        assert len(path_list[0]) >= 2
        has_label = True if len(path_list[0]) == 4 else False
        source_path_list = [path_list[i][0] for i in range(num_pair)]
        target_path_list = [path_list[i][1] for i in range(num_pair)]
        l_source_path_list = None
        l_target_path_list = None
        if has_label:
            l_source_path_list = [path_list[i][2] for i in range(num_pair)]
            l_target_path_list = [path_list[i][3] for i in range(num_pair)]
        return source_path_list, target_path_list, l_source_path_list, l_target_path_list

    def process_img_pool(self):
        """img pool shoudl include following thing:
        img_label_path_dic:{img_name:{'img':img_fp,'label':label_fp,...}
        img_label_dic: {img_name:{'img':img_np,'label':label_np},......}
        pair_name_list:[[pair1_s,pair1_t],[pair2_s,pair2_t],....]
        pair_list [[s_np,t_np,sl_np,tl_np],....]
        only the pair_list need to be used by get_item method
        """
        img_label_path_dic = {}
        pair_name_list = []
        for fps in self.path_list:
            for i in range(2):
                fp = fps[i]
                fn = get_file_name(fp)
                if fn not in img_label_path_dic:
                    if self.has_label:
                        img_label_path_dic[fn] = {'img': fps[i], 'label': fps[i + 2]}
                    else:
                        img_label_path_dic[fn] = {'img': fps[i]}
            pair_name_list.append([get_file_name(fps[0]), get_file_name(fps[1])])
        split_dict = self.__split_dict(img_label_path_dic, num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=self.sub_process, args=(split_dict[i],))
            p.start()
            print("pid:{} start:".format(p.pid))

            procs.append(p)

        for p in procs:
            p.join()

        print("completed the processing in {}".format(self.phase))

    def sub_process(self, img_label_path_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(img_label_path_dic)).start()
        count = 0
        for _, img_label_path in img_label_path_dic.items():
            img_pth = img_label_path['img']
            label_pth = img_label_path['label']
            self.resize_input_img_and_save_it(img_pth, is_label=False, fname=get_file_name(img_pth))
            self.resize_input_img_and_save_it(label_pth, is_label=True, fname=get_file_name(label_pth))
            count += 1
            pbar.update(count)
        pbar.finish()

    def convert_label_map_into_standard_one(self, label):
        label_np = sitk.GetArrayFromImage(label)
        cur_label = list(np.unique(label_np))
        extra_label = set(cur_label) - self.shared_label_set
        # print(" the extra label is {}. ".format(extra_label))
        if len(extra_label) == 0:
            print("no extra label")
        for elm in extra_label:
            """here we assume the value 0 is the background"""
            label_np[label_np == elm] = 0
        label = sitk.GetImageFromArray(label_np)
        return label

    def resize_input_img_and_save_it(self, img_pth, is_label=False, fname='', keep_physical=True):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        img_org = sitk.ReadImage(img_pth)
        img = self.__read_and_clean_itk_info(img_pth)
        if is_label and self.shared_label_set is not None:
            img = self.convert_label_map_into_standard_one(img)
        dimension = 3
        factor = np.flipud(self.resize_factor)
        img_sz = img.GetSize()
        if self.resize:
            resampler = sitk.ResampleImageFilter()

            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [int(img_sz[i] * factor[i]) for i in range(dimension)]
            after_size = [int(sz) for sz in after_size]
            matrix[0, 0] = 1. / factor[0]
            matrix[1, 1] = 1. / factor[1]
            matrix[2, 2] = 1. / factor[2]
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
        output_path = self.data_output_path
        output_path = os.path.join(output_path, 'img') if not is_label else os.path.join(output_path, 'label')
        os.makedirs(output_path, exist_ok=True)
        fpth = os.path.join(output_path, fname + '.nii.gz')
        if keep_physical:
            img_resampled.SetSpacing(resize_spacing(img_sz, img_org.GetSpacing(), factor))
            img_resampled.SetOrigin(img_org.GetOrigin())
            img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth

    def __read_and_clean_itk_info(self, path):
        if path is not None:
            return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))
        else:
            return None

    def __split_dict(self, dict_to_split, split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), num_of_workers)
        split_dict = []
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj = dict(dict_to_split_items[index_split[i][0]:index_split[i][-1] + 1])
            split_dict.append(dj)
        return split_dict

    def __inverse_name(self, name):
        try:
            n_parts = name.split('_image_')
            inverse_name = n_parts[1] + '_' + n_parts[0] + '_image'
            return inverse_name
        except:
            n_parts = name.split('_brain_')
            inverse_name = n_parts[1] + '_' + n_parts[0] + '_brain'
            return inverse_name

    def __len__(self):
        return len(self.name_list)  #############################3


data_path = '/playpen-raid/zyshen/data/reg_debug_labeled_oai_reg_inter'
phase_list = ['test']
""" path for saving the pair_path_list, pair_name_list"""
task_output_path = '/playpen-raid/zyshen/for_llf/reg_debug_labeled_oai_reg_inter'#'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
""" path for where to read the image during running the tasks"""
running_read_path = '/pine/scr/z/y/zyshen/reg_debug_labeled_oai_reg_inter/data'#'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
""" path for where to save the data"""
data_output_path = '/playpen-raid/zyshen/oai_data/reg_debug_labeled_oai_reg_inter/data'
""" img path need to be replaced with running_read_img_path"""
real_img_path = '/playpen-raid/zyshen/oai_data/Nifti_rescaled' #'/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled'
""" label path need to be replaced with runing_read_label_path """
real_label_path = ['/playpen-raid/zyshen/oai_data/Nifti_rescaled']
    #['/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038',
    #               '/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_left/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038']
resize_factor = [1,1,1]#[80./160.,192./384.,192./384]
shared_label_set=None



# data_path = '/playpen-raid/zyshen/data/reg_debug_3000_pair_oai_reg_intra'
# phase_list = ['train','val','debug']
# """ path for saving the pair_path_list, pair_name_list"""
# task_output_path = '/playpen-raid/zyshen/data/reg_debug_3000_pair_oai_reg_intra'
# """ path for where to read the image during running the tasks"""
# running_read_path = '/playpen-raid/zyshen/oai_data/reg_debug_labeled_oai_reg_intra/data'
# """ path for where to save the data"""
# data_output_path = '/playpen-raid/zyshen/oai_data/reg_debug_labeled_oai_reg_intra/data'
# """ img path need to be replaced with running_read_img_path"""
# real_img_path = '/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled'
# """ label path need to be replaced with runing_read_label_path """
# real_label_path = ['/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038',
#                    '/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_left/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038']
# resize_factor = [80./160.,192./384.,192./384]
# shared_label_set=None


# data_path = '/playpen-raid1/zyshen/data/reg_oai_aug'
# phase_list = ['train','val','debug','teset']
# """ path for saving the pair_path_list, pair_name_list"""
# task_output_path = '/playpen-raid1/zyshen/data/reg_oai_aug'
# """ path for where to read the image during running the tasks"""
# running_read_path = '/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_intra'
# """ path for where to save the data"""
# data_output_path = '/playpen-raid/zyshen/oai_data/reg_debug_labeled_oai_reg_intra/data'
# """ img path need to be replaced with running_read_img_path"""
# real_img_path = '/playpen-raid/zyshen/oai_data/Nifti_rescaled'#'/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled'
# """ label path need to be replaced with runing_read_label_path """
# real_label_path = ['/playpen-raid/zyshen/oai_data/Nifti_rescaled']
# #['/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038',
# #                    '/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_left/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038']
# resize_factor = [80./160.,192./384.,192./384]
# shared_label_set=None



#
# data_path = '/playpen-raid/zyshen/data/reg_debug_3000_pair_reg_224_oasis3_reg_inter'
# phase_list = ['train','val','debug','test']
# """ path for saving the pair_path_list, pair_name_list"""
# task_output_path = '/playpen-raid/zyshen/data/croped_lfix_for_reg_debug_3000_pair_reg_224_oasis3_reg_inter'#'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
# """ path for where to read the image during running the tasks"""
# running_read_path = '/playpen-raid/zyshen/oasis_data/croped_lfix_for_reg_debug_3000_pair_reg_224_oasis3_reg_inter/data'
#     #'/pine/scr/z/y/zyshen/croped_for_reg_debug_3000_pair_oai_reg_inter/data'#'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
# """ path for where to save the data"""
# data_output_path = '/playpen-raid/zyshen/oasis_data/todel/data'
# """ img path need to be replaced with running_read_img_path"""
# real_img_path = '/playpen-raid/xhs400/OASIS_3/processed_images_centered_224_224_224'
# """ label path need to be replaced with runing_read_label_path """
# real_label_path = ['/playpen-raid/xhs400/OASIS_3/processed_images_centered_224_224_224']
# resize_factor = [112./224.,112./224.,112./224]
# """Attention, here we manually add id 62 into list, for it is a big structure and is not absent in val, debug, test dataset"""
# #shared_label_set =  {0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 80, 85, 251, 252, 253, 254, 255}
# shared_label_set =  {0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63, 77}
#
# #
#
#
# data_path = '/playpen-raid/zyshen/data/syn_data'
# phase_list = ['test']
# """ path for saving the pair_path_list, pair_name_list"""
# task_output_path = '/playpen-raid/zyshen/for_llf/syn_2d'  # '/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
# """ path for where to read the image during running the tasks"""
# running_read_path = '/pine/scr/z/y/zyshen/data/syn_data/syn_2d/data'  # '/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter'
# """ path for where to save the data"""
# data_output_path = '/playpen-raid/zyshen/syn_data/2d_syn/data'
# """ img path need to be replaced with running_read_img_path"""
# real_img_path = '/playpen-raid/zyshen/debugs/syn_expr_0422_2'  # '/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled'
# """ label path need to be replaced with runing_read_label_path """
# real_label_path = ['/playpen-raid/zyshen/debugs/syn_expr_0422_2']
# # ['/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038',
# #               '/playpen-raid/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_left/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038']
# resize_factor = [1, 1, 1]  # [80./160.,192./384.,192./384]
# shared_label_set = None
#
for phase in phase_list:
    dataset = RegistrationDataset(data_path=os.path.join(data_path, phase), phase=phase, resize_factor=resize_factor)
    dataset.set_task_output_path(os.path.join(task_output_path, phase))
    dataset.set_data_output_path(os.path.join(data_output_path, phase))
    dataset.set_real_data_path(real_img_path, real_label_path)
    dataset.set_running_read_path(os.path.join(running_read_path, phase))

    if shared_label_set is not None:
        dataset.set_shared_label(shared_label_set)
    dataset.process()


