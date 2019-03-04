import SimpleITK as sitk
import numpy as np
import glob


import sys
import os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))

from functools import reduce
from data_pre.reg_data_utils import make_dir
from multiprocessing import Pool


def get_bounding_box(label):
    """
    given a label mask, return the bounding box of non-background objects
    :param label: sitk 3d image object
    :return: bounding mask: numpy array in shape (1,6) [starting_position(x,y,z) end_position(x,y,z)]
    """
    label_shape_stat = sitk.LabelShapeStatisticsImageFilter()  # filter to analysis the label shape
    label_shape_stat.Execute(label)
    box = np.array(label_shape_stat.GetBoundingBox(1), ndmin=2)
    box[0,3:] = box[0,3:]+box[0,:3]
    return box
def remove_abnormal_data(file_path_list):
    """
    the function would return file_boundary_list, each element include a numpy include boundary_distance [X_top,X_bottom, Y_top, Y_bottom, Z_top,Z_bottom]
    :param file_path_list:
    :return:
    """
    boundary_list = []

    for fp in file_path_list:
        img = sitk.ReadImage(fp)
        boundary = get_bounding_box(img>0)
        boundary_list.append(boundary)
    return boundary_list



f = remove_abnormal_data




#from data_pre.reg_data_utils import *

# all_imgs = read_txt_into_list('/playpen/zyshen/data/reg_debug_3000_pair_oasis_reg_inter/train/pair_path_list.txt')
# all_imgs = [img_pth[0] for img_pth in all_imgs] +[img_pth[1] for img_pth in all_imgs]
all_imgs = glob.glob('/playpen/xhs400/OASIS_3/processed_images/*brain.nii.gz')
print(len(all_imgs))


number_of_workers = 20
boundary_list=[]
file_patitions = np.array_split(all_imgs, number_of_workers)
with Pool(processes=number_of_workers) as pool:
    boundary = pool.map(f, file_patitions)
for fp_list in boundary:
    for fp in fp_list:
        boundary_list.append(fp)
boundarys = np.array(boundary_list)
min_boundary = np.min(boundarys[:,0,:3],axis=0)
min_boundary_index = np.argmin(boundarys[:,0,:3],axis=0)
max_boundary = np.max(boundarys[:,0,3:],axis=0)
max_boundary_index = np.argmin(boundarys[:,0,3:],axis=0)
print("complete the boundary detection")
print("the max distance can be crop is {},{}".format(min_boundary,224-max_boundary))
print("--------------- for the left up front the corresponindg image is-------------------")
print('{},\n{},\n{}'.format(all_imgs[min_boundary_index[0]],all_imgs[min_boundary_index[1]],all_imgs[min_boundary_index[2]]))
print("--------------- for the right down behind the corresponindg image is-------------------")
print('{},\n{},\n{}'.format(all_imgs[max_boundary_index[0]],all_imgs[max_boundary_index[1]],all_imgs[max_boundary_index[2]]))
#
#
# crop_image_filter =sitk.CropImageFilter()
# crop_image_filter.SetUpperBoundaryCropSize([16,16,16])
# crop_image_filter.SetLowerBoundaryCropSize([16,16,16])
#
# for img_path in all_imgs:
# #    if 'OAS30041_MR_d6548' not in img_path:
# #        continue
#     print(img_path)
#     lbl_path = img_path.replace('brain', 'label')
#     print(lbl_path)
#
#     img = sitk.ReadImage(img_path)
#     cropped_image = sitk.Cast(crop_image_filter.Execute(img), sitk.sitkFloat32)
#
#     lbl = sitk.ReadImage(lbl_path)
#     cropped_label = crop_image_filter.Execute(lbl)
#
#
#     img_arr = sitk.GetArrayFromImage(img)
#     intensities = img_arr.reshape(-1)
#     i_99 = np.percentile(intensities, 99)
#
#     img_norm_1 = sitk.ShiftScale(image1=cropped_image, shift=0, scale=0.99/i_99)
#     img_norm_2 = sitk.IntensityWindowing(img_norm_1, windowMinimum=0.0, windowMaximum=1.0, outputMinimum=0.0, outputMaximum=1.0)
#
#     img_file = img_path.split('/')[2]
#     lbl_file = img_file.replace('brain', 'label')
#     print(img_file)
#     print(lbl_file)
#
#     sitk.WriteImage(img_norm_2, img_file)
#     sitk.WriteImage(cropped_label, lbl_file)
