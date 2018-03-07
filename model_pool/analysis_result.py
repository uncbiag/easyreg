import numpy as np
from glob import glob
import os
def get_file_list(post_type,data_path, file_mid_tag,file_end_tag,gt_tag):
    f_filter = []
    f_path = os.path.join(data_path, '**', file_mid_tag+post_type)
    f_filter = glob(f_path, recursive=True)
    fname_full_list = [os.path.split(file)[1].split(file_end_tag)[0] for file in f_filter]
    fname_set = set(fname_full_list)
    for fname in fname_set:
        f_path = os.path.join(data_path, '**', fname+file_mid_tag+post_type)
        f_filter = glob(f_path, recursive=True)

    return f_filter
    # sample_label_path = sample_data_path
    # file_name = file_name.split('_t')[0]
    # from os.path import join
    # label_post = os.path.split(sample_label_path)[1].split('.', 1)[1]
    # f_path = join(raw_data_path, '**', file_name + '.' + label_post)
    # f_filter = glob(f_path, recursive=True)
    # if len(f_filter) == 1:
    #     image = sitk.ReadImage(f_filter[0])
    # else:
    #     print("Warning, the source file is not founded during file saving, default info from {} is used".format(
    #         sample_label_path))
    #     sample_label_path = self.opt['tsk_set']['extra_info']['sample_label_path']
    #     image = sitk.ReadImage(sample_label_path)
    # self.origin_size = sitk.GetArrayFromImage(image).shape


# def get_period_voting_map(period_record_dic):
#     """
#     :param period_record_dic:  dict, include the filename:(ith_output_map, ith_iter)
#     :return: dic, include the filename: voting_map
#     """
#     for item in period_record_dic:
#         multi_period_res = [period_res[0] for period_res in item]
#         multi_period_res = np.stack(multi_period_res,0)
#         period_voting_map = np.max(multi_period_res,0)


post_type = '*_output.nii.gz'
data_path ='/playpen/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_106_unet_resid_only/records/output'
file_end_tag ='_t'
gt_tag = '*_gt.nii.gz'
file_mid_tag ='*_val_'
files = get_file_list(post_type, data_path,file_mid_tag, file_end_tag,gt_tag)
