import numpy as np
from glob import glob
import os
import SimpleITK as sitk

from data_pre.seg_data_utils import make_dir
from model_pool.metrics import get_multi_metric


def get_file_list(post_type,data_path, file_mid_tag,file_end_tag,gt_tag,show_period_result=True, debug=False):
    f_path = os.path.join(data_path, '**', '*'+file_mid_tag+'*'+post_type)
    f_filter = glob(f_path, recursive=True)
    fname_full_list = [os.path.split(file)[1].split(file_end_tag)[0] for file in f_filter]
    fname_set = set(fname_full_list)
    saving_folder_path = os.path.join(os.path.split(data_path)[0], 'voting')
    make_dir(saving_folder_path)
    for fname in fname_set:
        txt_path = saving_folder_path + '/' + fname+'.txt'
        if not debug:
            text_file = open(txt_path, "w")
        else:
            text_file = None
        f_path = os.path.join(data_path, '**', fname+'*'+file_mid_tag+'*'+post_type)
        f_filter = glob(f_path, recursive=True)
        if len(f_filter):
            f_label_path = f_filter[0].replace(post_type, gt_tag)
            gt_itk = sitk.ReadImage(f_label_path)
            gt =sitk.GetArrayFromImage(gt_itk)
            label_list = np.unique(gt)
        else:
            print("Warning, no label was founded when dealing with image {}".format(fname), file=text_file)
            break
        imgs=[]
        print('****************************************   period result  ************************************************************')

        for f in f_filter:
            image = sitk.ReadImage(f)
            image=sitk.GetArrayFromImage(image)
            imgs +=[image]
            file_name= os.path.split(f)[1]
            print("the current image period is {}".format(file_name), file=text_file)
            val_res_dic = get_multi_metric(np.expand_dims(image, 0), np.expand_dims(gt,0),
                                           rm_bg=False)
            print("the result :", file=text_file)
            print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']['dice']), file=text_file)
            print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']['dice']), file=text_file)
        imgs = np.stack(imgs,0)

        print('#########################################  voting   ########################################################################')
        print("the image {} has {} period record".format(fname, imgs.shape[0]), file=text_file)

        for i in range(imgs.shape[0]):
            period_voting_map = cal_voting_map(imgs[i:], label_list)
            val_res_dic = get_multi_metric(np.expand_dims(period_voting_map, 0),  np.expand_dims(gt,0),
                                           rm_bg=False)  # the correct version maybe np.expand_dims(np.expand_dims(self.output,0))
            print('the voting result of file{}  from period  {}:'.format(fname,i), file=text_file)
            print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']['dice']), file=text_file)
            print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']['dice']), file=text_file)
            print()

            period_voting_map = sitk.GetImageFromArray(period_voting_map)
            period_voting_map.CopyInformation(gt_itk)
            appendix = fname+'_from_period_'+str(i) + "_voting"
            saving_file_path = saving_folder_path + '/' + appendix + "_output.nii.gz"
            sitk.WriteImage(period_voting_map, saving_file_path)
        if not debug:
            text_file.close()

    return f_filter

def cal_voting_map(multi_period_map, label_list):
    count_map = np.zeros([len(label_list)]+ list(multi_period_map.shape)[1:])

    for i, label in enumerate(label_list):
        count_map[i] = np.sum(multi_period_map==label,0)
    period_voting_map = np.argmax(count_map, 0)
    period_voting_map_re = np.zeros_like(period_voting_map)

    for i, label in enumerate(label_list):
        period_voting_map_re[np.where(period_voting_map == i)] = label
    return period_voting_map_re


post_type = '_output.nii.gz'
data_path ='/playpen/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_106_unet_resid_only/records/output'
file_end_tag ='_t'
gt_tag = '_gt.nii.gz'
file_mid_tag ='_val_'
files = get_file_list(post_type, data_path,file_mid_tag, file_end_tag,gt_tag,debug=True)
