
import numpy as np
import os
import SimpleITK as sitk
import glob

def compute_jacobi_map(jacobian,crop_boundary=True):
    if crop_boundary:
        crop_range = 10
        jacobian = jacobian[crop_range:-crop_range, crop_range:-crop_range, crop_range:-crop_range]
    jacobi_abs = np.sum(jacobian)  #
    jacobi_num = np.sum(jacobian>0.)
    print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
    print("the number of fold points for current batch is {}".format(jacobi_num))
    # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
    jacobi_abs_mean = jacobi_abs  # / np.prod(map.shape)
    return jacobi_abs_mean,jacobi_num





def get_mermaid_jacobi(fpath,mask_path):
    jacobi_image = sitk.ReadImage(os.path.join(fpath))
    jacobi_mask = sitk.ReadImage(os.path.join(mask_path))
    jacobi_np = sitk.GetArrayFromImage(jacobi_image)
    jacobi_np = sitk.GetArrayFromImage(jacobi_mask)*jacobi_np
    return jacobi_np

def compute_jacobi(record_path):

    f_list =glob.glob(os.path.join(record_path,'**/*_0000jacobi_img.nii'), recursive=True)
    f_mask_list =[f.replace('_0000jacobi_img.nii','_0000jacobi_neg_img.nii') for f in f_list]
    num_samples = len(f_list)
    records_jacobi_val_np = np.zeros(num_samples)
    records_jacobi_num_np = np.zeros(num_samples)
    jacobi_val_res = 0.
    jacobi_num_res = 0.
    for i, f_path in enumerate(f_list):
        batch_size=1
        jacobian_np = get_mermaid_jacobi(f_path,f_mask_list[i])
        extra_res = compute_jacobi_map(jacobian_np)
        jacobi_val_res += extra_res[0] * batch_size
        jacobi_num_res += extra_res[1] * batch_size
        records_jacobi_val_np[i] = extra_res[0]
        records_jacobi_num_np[i] = extra_res[1]
        print("id {} and current pair name is : {}".format(i,os.path.split(f_path)[-1]))
        print('the current averge jocobi val is {}'.format(jacobi_val_res/(i+1)/batch_size))
        print('the current averge jocobi num is {}'.format(jacobi_num_res/(i+1)/batch_size))
    jacobi_val_res = jacobi_val_res/num_samples
    jacobi_num_res = jacobi_num_res/num_samples
    print("the average {}_ jacobi val: {}  :".format('test', jacobi_val_res))
    print("the average {}_ jacobi num: {}  :".format('test', jacobi_num_res))
    np.save(os.path.join(record_path,'croped_records_jacobi'),records_jacobi_val_np)
    np.save(os.path.join(record_path,'croped_records_jacobi_num'),records_jacobi_num_np)



record_path = '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/reg_adpt_lddamm_wkw_formul_05_1_omt_2step_200sym_minstd_005_sm008_allinterp_maskv_epdffix/reg/res'
compute_jacobi(record_path)