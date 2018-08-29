import sys
import subprocess
import os

from model_pool.nifty_reg_utils import *


mv_path = ''
target_path = ''
registration_type = 'affine'
moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled/9002116_20050715_SAG_3D_DESS_RIGHT_10423916_image.nii.gz'
target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_6sets_rescaled/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_image.nii.gz'
moving_label_path ='/playpen/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/' \
                   'Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038/' \
                   '9002116_20050715_SAG_3D_DESS_RIGHT_10423916_prediction_step1_batch6_16_reflect.nii.gz'
performRegistration(moving_img_path,target_img_path,registration_type,record_path=None,ml_path=moving_label_path)
disp = nifty_read('./displacement.nii')
deform = nifty_read('./deformation.nii') #we need to use

print("done")
