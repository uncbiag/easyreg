import numpy as np
from glob import glob
import os
import SimpleITK as sitk
import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
from data_pre.seg_data_utils import make_dir
from model_pool.metrics import get_multi_metric
from multiprocessing import Pool, TimeoutError
#from .vonet_pool import UNet_asm
import matplotlib.pyplot as plt
import torch



def get_file_list(data_path,post_type, file_mid_tag,file_end_tag,gt_tag,show_period_result=True, debug=False):
    if isinstance(data_path, (np.ndarray)):
        for path in data_path:
            get_file_list(path,post_type, file_mid_tag, file_end_tag, gt_tag, show_period_result=True,
                          debug=debug)
        return 0

    print("processing {}".format(data_path))
    f_path = os.path.join(data_path, '**', '*'+file_mid_tag+'*'+post_type)
    f_filter = glob(f_path, recursive=True)
    if len(f_filter):
        fname_full_list = [os.path.split(file)[1].split(file_end_tag)[0] for file in f_filter]
        fname_set = list(set(fname_full_list))
        saving_folder_path = os.path.join(os.path.split(data_path)[0], 'voting')
        make_dir(saving_folder_path)

        # file_patitions = np.array_split(fname_set,number_of_workers)
        # from functools import partial
        # with Pool(processes=number_of_workers) as pool:
        #     pool.map(partial(period_analysis,saving_folder_path=saving_folder_path,post_type=post_type,data_path=data_path, file_mid_tag=file_mid_tag,gt_tag=gt_tag, debug=debug), file_patitions)
        period_analysis(fname_set,saving_folder_path,post_type,data_path, file_mid_tag,gt_tag, debug)
    else:
        print('there is no valid result in {}'.format(data_path))


def period_analysis(fname_set,saving_folder_path,post_type,data_path, file_mid_tag,gt_tag, debug):
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
        print('****************************************   period result  ************************************************************',file=text_file)
        period_list = []
        period_res_list = []
        for f in f_filter:
            image = sitk.ReadImage(f)
            image=sitk.GetArrayFromImage(image)
            imgs +=[image]
            file_name= os.path.split(f)[1]
            val_period = file_name.split('iter_')[1].split(post_type)[0]
            period_list.append(val_period)
            file_name = fname+ file_name.split('iter')[1].split(post_type)[0]
            print("the current image period is {}".format(file_name), file=text_file)
            val_res_dic = get_multi_metric(np.expand_dims(image, 0), np.expand_dims(gt,0),
                                           rm_bg=False)
            period_res_list.append(val_res_dic['batch_label_avg_res']['dice'])
            print("the result :", file=text_file)
            print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']['dice']), file=text_file)
            print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']['dice']), file=text_file)
        imgs = np.stack(imgs,0)

        print('#########################################  voting   ########################################################################',file=text_file)
        print("the image {} has {} period record".format(fname, imgs.shape[0]), file=text_file)
        period_ens_voting_list = []
        for i in range(imgs.shape[0]):
            period_voting_map = cal_voting_map(imgs[i:], label_list)
            val_res_dic = get_multi_metric(np.expand_dims(period_voting_map, 0),  np.expand_dims(gt,0),
                                           rm_bg=False)  # the correct version maybe np.expand_dims(np.expand_dims(self.output,0))
            period_ens_voting_list.append(val_res_dic['batch_label_avg_res']['dice'])
            print('the voting result of file{}  from period  {}:'.format(fname,i), file=text_file)
            print('batch_avg_res{}'.format(val_res_dic['batch_avg_res']['dice']), file=text_file)
            print('batch_label_avg_res:{}'.format(val_res_dic['batch_label_avg_res']['dice']), file=text_file)
            print(file=text_file)

            period_voting_map = sitk.GetImageFromArray(period_voting_map)
            period_voting_map.CopyInformation(gt_itk)
            appendix = fname+'_from_period_'+str(i) + "_voting"
            saving_file_path = saving_folder_path + '/' + appendix + "_output.nii.gz"
            sitk.WriteImage(period_voting_map, saving_file_path)
        plot_res(period_res_list,period_ens_voting_list,period_list,fname,saving_folder_path)
        if not debug:
            text_file.close()

def cal_voting_map(multi_period_map, label_list):
    count_map = np.zeros([len(label_list)]+ list(multi_period_map.shape)[1:])

    for i, label in enumerate(label_list):
        count_map[i] = np.sum(multi_period_map==label,0)
    period_voting_map = np.argmax(count_map, 0)
    period_voting_map_re = np.zeros_like(period_voting_map)

    for i, label in enumerate(label_list):
        period_voting_map_re[np.where(period_voting_map == i)] = label
    return period_voting_map_re




def plot_res(period_res_list, period_ens_voting_list, period_list, fname, saving_path):
    plt.figure(figsize=( 9.,3.841), dpi=300)
    plt.plot(range(len(period_res_list)), period_res_list, label="single_period")
    plt.plot(range(len(period_ens_voting_list)), period_ens_voting_list, label="ensemble_period")
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xticks(range(len(period_res_list)), period_list)
    plt.ylabel('dice')
    plt.ylabel('step')
    plt.title(fname)
    plt.savefig(os.path.join(saving_path,fname+'.png'),dpi=300)
    #plt.show()
    #plt.draw()
    plt.clf()





#
# post_type = '_output.nii.gz'
# data_path ='/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_105_unet/records/output'
# file_end_tag ='_t'
# gt_tag = '_gt.nii.gz'
# file_mid_tag ='_val_'
# number_of_workers=10
# get_file_list(data_path,post_type,file_mid_tag, file_end_tag,gt_tag,debug=False)
#







# post_type = '_output.nii.gz'
# #data_path ='/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_106_unet_resid_only/records/output'
# data_path ='/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_105_unet/records/output'
# file_end_tag ='_t'
# gt_tag = '_gt.nii.gz'
# file_mid_tag ='_val_'
# number_of_workers=10
# #get_file_list(data_path,post_type,file_mid_tag, file_end_tag,gt_tag,debug=False)
#
# root_path = "/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop"
# sub_dirs = next(os.walk(root_path))[1]
# key_word_list = ['task','tsk']
# for sub_dir in sub_dirs:
#     has_task= sum([(key_word in sub_dir) for key_word in key_word_list])
#     if has_task:
#         record_path = root_path +'/'+sub_dir +'/'+'records/output'
#         if os.path.isdir(record_path):
#             print(record_path)
#             get_file_list(record_path,post_type,file_mid_tag, file_end_tag,gt_tag,debug=False)



post_type = '_output.nii.gz'
#data_path ='/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_106_unet_resid_only/records/output'
#data_path ='/playpen/raid/zyshen/data/brats_com_brats_seg_patchedmy_balanced_random_crop/tsk_105_unet/records/output'
file_end_tag ='_t'
gt_tag = '_gt.nii.gz'
file_mid_tag ='_val_'
number_of_workers=20
root_path = "/playpen/zyshen/data/oai_2_vnet_oai_seg_nopatchedmy_balanced_random_crop"
sub_dirs = next(os.walk(root_path))[1]
key_word_list = ['task','tsk']
valid_record_path = []
for sub_dir in sub_dirs:
    has_task= sum([(key_word in sub_dir) for key_word in key_word_list])
    if has_task:
        record_path = root_path +'/'+sub_dir +'/'+'records/output'
        if os.path.isdir(record_path):
            valid_record_path +=[record_path]
record_path_patitions = np.array_split(valid_record_path, number_of_workers)
from functools import partial
with Pool(processes=number_of_workers) as pool:
    pool.map(partial(get_file_list,post_type=post_type,file_mid_tag=file_mid_tag, file_end_tag=file_end_tag,gt_tag=gt_tag,debug=False), record_path_patitions)
