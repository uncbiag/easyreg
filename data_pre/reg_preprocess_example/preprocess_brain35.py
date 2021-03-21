"""
A demo on data augmentation and segmentation for brain dataset
"""


import os, sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../..'))
sys.path.insert(0,os.path.abspath('.'))
from data_pre.file_tool import get_file_list
from  easyreg.reg_data_utils import read_txt_into_list, write_list_into_txt
from data_pre.seg_data_pool import BaseSegDataSet


def find_corr_label(img_path_list,label_root_path=None,label_switch=None):
    get_par_folder_name = lambda x: os.path.split(os.path.split(os.path.split(x)[0])[0])[-1]
    fname_list = [get_par_folder_name(path) for path in img_path_list]
    label_path_list = [get_file_list('/playpen-raid1/Data/annotation',fname+"*"+".nii.gz")[0] for fname in fname_list]
    if label_root_path is not None:
        label_path_list = [path.replace(os.path.split(path)[0],label_root_path) for path in label_path_list]
    return label_path_list

def get_file_name( img_path):
    get_par_folder_path = lambda x: os.path.split(os.path.split(x)[0])[0]
    file_name = os.path.split(get_par_folder_path(img_path))[-1]
    return file_name

dataset = BaseSegDataSet(file_type_list=["T1w_acpc_dc_restore.nii.gz"])
data_path = "/playpen-raid1/Data/Brain35"
output_path ='/playpen-raid1/zyshen/data/brain_35/non_resize'
divided_ratio = (0.6,0.1,0.3)
dataset.set_data_path(data_path)
dataset.find_corr_label = find_corr_label
dataset.get_file_name = get_file_name
dataset.set_output_path(output_path)
dataset.set_divided_ratio(divided_ratio)
dataset.img_after_resize = (260,311,260)
#dataset.prepare_data()



from easyreg.aug_utils import gen_post_aug_pair_list
train_file_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_path_list.txt"
test_file_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt"
train_name_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_name_list.txt"
test_name_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_name_list.txt"
output_file_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/test_aug_path_list.txt"
output_name_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/test_aug_name_list.txt"


train_path_list = read_txt_into_list(train_file_path)
test_path_list = read_txt_into_list(test_file_path)
train_name_list = read_txt_into_list(train_name_path)
test_name_list = read_txt_into_list(test_name_path)
test_img_path_list = [path[0] for path in test_path_list]
test_label_path_list = [path[1] for path in test_path_list]
if isinstance(train_path_list[0],list):
    train_img_path_list = [path[0] for path in train_path_list]
    train_label_path_list = [path[1] for path in train_path_list]
else:
    train_img_path_list = train_path_list
    train_label_path_list = None

img_pair_list, pair_name_list = gen_post_aug_pair_list(test_img_path_list,train_img_path_list, test_fname_list=test_name_list,train_fname_list=train_name_list,
                           test_label_path_list=test_label_path_list,train_label_path_list=train_label_path_list, pair_num_limit=-1, per_num_limit=5)
pair_name_list = [pair_name[1:] for pair_name in pair_name_list]
# write_list_into_txt(output_file_path,img_pair_list)
# write_list_into_txt(output_name_path,pair_name_list)



train_aug_output_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train"
train_aug_output_full_path = train_aug_output_path+"/aug"
output_folder = "/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k2/train"
os.makedirs(output_folder,exist_ok=True)
output_path = os.path.join(output_folder,"file_path_list.txt")
train_aug_img_list = get_file_list(train_aug_output_full_path,"*_image.nii.gz")
train_aug_label_list = [path.replace("_image.nii.gz","_label.nii.gz") for path in train_aug_img_list]
img_label_path_list = [[img_path, label_path] for img_path, label_path in zip(train_aug_img_list,train_aug_label_list)]
write_list_into_txt(output_path,img_label_path_list)
# #
# train_aug_output_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train_random"
# train_aug_output_full_path = train_aug_output_path+"/aug"
# output_folder = "/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_random/train"
# os.makedirs(output_folder,exist_ok=True)
# output_path = os.path.join(output_folder,"file_path_list.txt")
# train_aug_img_list = get_file_list(train_aug_output_full_path,"*_image.nii.gz")
# train_aug_label_list = [path.replace("_image.nii.gz","_label.nii.gz") for path in train_aug_img_list]
# img_label_path_list = [[img_path, label_path] for img_path, label_path in zip(train_aug_img_list,train_aug_label_list)]
# write_list_into_txt(output_path,img_label_path_list)
#
#
#
# train_aug_output_path = "/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train_bspline"
# train_aug_output_full_path = train_aug_output_path+"/aug"
# output_folder = "/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_bspline/train"
# os.makedirs(output_folder,exist_ok=True)
# output_path = os.path.join(output_folder,"file_path_list.txt")
# train_aug_img_list = get_file_list(train_aug_output_full_path,"*_image.nii.gz")
# train_aug_label_list = [path.replace("_image.nii.gz","_label.nii.gz") for path in train_aug_img_list]
# img_label_path_list = [[img_path, label_path] for img_path, label_path in zip(train_aug_img_list,train_aug_label_list)]
# write_list_into_txt(output_path,img_label_path_list)


"""
training phase augmentation
python demo_for_data_aug.py --file_txt=/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_path_list.txt --name_txt=/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_name_list.txt --txt_format=aug_by_file --setting_folder_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_brain35 --task_output_path=/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train --gpu_id_list 0 1 2 3 0 1 2 3

testing phase augmentation
python demo_for_data_aug.py --file_txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test_aug_path_list.txt --name_txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test_aug_name_list.txt --txt_format=aug_by_line --setting_folder_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_brain35_postaug --task_output_path=/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_test --gpu_id_list 0 1


training phase augmentation (random)
python gen_aug_samples.py -t=/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_path_list.txt -as=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/rand_lddmm_brain35_random/data_aug_setting.json -ms=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/rand_lddmm_brain35_random/mermaid_nonp_settings.json -o=/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train_random/aug -g=2

testing phase augmentation (random)
python gen_aug_samples.py -t=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt -as=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/rand_lddmm_brain35_postaug_random/data_aug_setting.json -ms=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/rand_lddmm_brain35_random/mermaid_nonp_settings.json -o=/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_test_random/aug -g=1


training phase augmentation (bspline)
python gen_aug_samples.py -t=/playpen-raid1/zyshen/data/brain_35/non_resize/train/file_path_list.txt -as=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/rand_bspline_brain35/data_aug_setting.json --bspline -o=/playpen-raid1/zyshen/data/brain_35/non_resize/data_aug_train_bspline/aug



train segmentation without aug
python demo_for_seg_train.py -o /playpen-raid1/zyshen/data/brain_35 -dtn=non_resize -tn=custom_seg_norm_intensity -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train -g=0


train segmentation with aug K=1
python demo_for_seg_train.py -o /playpen-raid1/zyshen/data/brain_35/non_resize -dtn=seg_aug_train_k1 -tn=aug_seg_norm_intensity -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k1 -g=1

train segmentation with aug K=2
python demo_for_seg_train.py -o /playpen-raid1/zyshen/data/brain_35/non_resize -dtn=seg_aug_train_k2 -tn=aug_seg_norm_intensity -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k2 -g=1

train segmentation with aug random
python demo_for_seg_train.py -o /playpen-raid1/zyshen/data/brain_35/non_resize -dtn=seg_aug_train_random -tn=aug_seg_norm_intensity -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_random -g=2

train segmentation with aug bspline
python demo_for_seg_train.py -o /playpen-raid1/zyshen/data/brain_35/non_resize -dtn=seg_aug_train_bspline -tn=aug_seg_norm_intensity -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_bspline -g=3


test segmentation without aug
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/custom_seg_norm_intensity/checkpoints/model_best.pth.tar  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/custom_seg_res -g=0


test segmentation with training aug k1
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k1 -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k1/aug_seg_norm_intensity/checkpoints/model_best.pth.tar  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k1 -g=1


test segmentation with training aug k2
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k2 -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k2/aug_seg_norm_intensity/checkpoints/model_best.pth.tar  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k2 -g=1


test segmentation with training_random aug
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_random -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_random/aug_seg_norm_intensity/checkpoints/model_best.pth.tar  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_random_res -g=2

test segmentation with bspline aug
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_bspline -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_bspline/aug_seg_norm_intensity/checkpoints/model_best.pth.tar  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_bspline_res -g=3


test segmentation with training testing aug k1
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k1 -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k1/aug_seg_norm_intensity/checkpoints/epoch_110_  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_and_test_res_trainedk1testk1 -g=0

test segmentation with training testing aug k2
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_k2 -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_k2/aug_seg_norm_intensity/checkpoints/epoch_150_  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_and_test_res_trainedk2testk2 -g=1

test segmentation with training testing random_aug
python demo_for_seg_eval.py -ts=/playpen-raid/zyshen/reg_clean/debug/brain35/seg_train_aug_random -txt=/playpen-raid1/zyshen/data/brain_35/non_resize/test/file_path_list.txt  -m=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_random/aug_seg_norm_intensity/checkpoints/epoch_150_  -o=/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_and_test_random_res -g=2

"""



