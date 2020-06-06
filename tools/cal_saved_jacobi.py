from mermaid import finite_differences as fdt
import numpy as np
import os
import SimpleITK as sitk
import glob
from easyreg.reg_data_utils import get_file_name


def compute_jacobi_map(map, spacing, fname_list=None, mask=None, crop_boundary=True, save_jacobi_map=False,saving_folder=""):
    """
    compute determinant jacobi on transformatiomm map,  the coordinate should be canonical.

    :param map: the transformation map
    :param crop_boundary: if crop the boundary, then jacobi analysis would only analysis on cropped map
    :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
    """
    fd = fdt.FD_np(spacing)
    dfx = fd.dXc(map[:, 0, ...])
    dfy = fd.dYc(map[:, 1, ...])
    dfz = fd.dZc(map[:, 2, ...])
    jacobi_det = dfx * dfy * dfz
    jacobi_det = jacobi_det.astype(np.float32)
    if mask is not None:
        jacobi_det = jacobi_det * mask
    average_jacobi_masked = -1

    if crop_boundary:
        crop_range = 5
        jacobi_det_croped = jacobi_det[:, crop_range:-crop_range, crop_range:-crop_range, crop_range:-crop_range]
        jacobi_abs_croped = - np.sum(jacobi_det_croped[jacobi_det_croped < 0.])  #
        jacobi_num_croped = np.sum(jacobi_det_croped < 0.)
        print("Cropped! the jacobi_value of fold points for current batch is {}".format(jacobi_abs_croped))
        print("Cropped! the number of fold points for current batch is {}".format(jacobi_num_croped))
        if mask is not None:
            mask_cropped = mask[ crop_range:-crop_range, crop_range:-crop_range, crop_range:-crop_range]
            average_jacobi_masked = np.sum(jacobi_det)/np.sum(mask_cropped)
            print("Cropped! the average jacobi value at the mask region is {}".format(average_jacobi_masked))

    # self.temp_save_Jacobi_image(jacobi_det,map)
    jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])  #
    jacobi_num = np.sum(jacobi_det < 0.)
    print("print folds for each channel {},{},{}".format(np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
    print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
    print("the number of fold points for current batch is {}".format(jacobi_num))
    jacobi_abs_mean = jacobi_abs / map.shape[0]
    jacobi_num_mean = jacobi_num / map.shape[0]
    if mask is not None:
        average_jacobi_masked = np.sum(jacobi_det) / np.sum(mask)
        print("the average jacobi value at the mask region is {}".format(average_jacobi_masked))
    
    
    jacobi_abs_map = np.abs(jacobi_det)
    if save_jacobi_map:
        jacobi_neg_map = np.zeros_like(jacobi_det)
        jacobi_neg_map[jacobi_det < 0] = 1
        for i in range(jacobi_abs_map.shape[0]):
            jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
            jacobi_neg_img = sitk.GetImageFromArray(jacobi_neg_map[i])
            jacobi_img.SetSpacing(np.flipud(spacing))
            jacobi_neg_img.SetSpacing(np.flipud(spacing))
            jacobi_saving = saving_folder
            os.makedirs(jacobi_saving, exist_ok=True)
            pth = os.path.join(jacobi_saving,
                               fname_list[i] + '_jacobi_img.nii.gz')
            n_pth = os.path.join(jacobi_saving,
                                 fname_list[i]+ '_jacobi_neg_img.nii.gz')
            sitk.WriteImage(jacobi_img, pth)
            sitk.WriteImage(jacobi_neg_img, n_pth)
    return jacobi_abs_mean, jacobi_num_mean, average_jacobi_masked





def read_deformation_and_mask(deformation_path, mask_path=None, itk_format=False):
    mask = None
    if mask_path is not None:
        mask = sitk.ReadImage(os.path.join(mask_path))
        mask = sitk.GetArrayFromImage(mask)

    if not itk_format:
        deformation = sitk.ReadImage(os.path.join(deformation_path))
        deformation_np = sitk.GetArrayFromImage(deformation)
        deformation_np = np.transpose(deformation_np,[3,2,1,0])[None]
        spacing = 1./(np.array(deformation_np.shape[2:])-1)
    else:
        deformation = sitk.ReadTransform(deformation_path)
        deformation_np = sitk.GetArrayFromImage(deformation.GetDisplacementField())[None]
        spacing = deformation_np.GetFixedParameters()[6:9]

    return deformation_np, mask, spacing



def compute_jacobi(deformation_path_list, fname_list, saving_folder="", mask_path_list=None, itk_format= False):
    num_samples = len(deformation_path_list)

    if mask_path_list is not None:
        assert len(deformation_path_list) == len(mask_path_list)
    else:
        mask_path_list = [None]*num_samples
    records_jacobi_val_np = np.zeros(num_samples)
    records_jacobi_num_np = np.zeros(num_samples)
    average_jacobi_masked_np = np.zeros(num_samples)
    jacobi_val_res = 0.
    jacobi_num_res = 0.
    average_jacobi_masked_res = 0.
    for i, f_path in enumerate(deformation_path_list):
        batch_size=1
        deformation_np, mask_np, spacing = read_deformation_and_mask(deformation_path_list[i], mask_path_list[i],itk_format)
        extra_res = compute_jacobi_map(deformation_np,spacing,[fname_list[i]],mask_np,crop_boundary=True,save_jacobi_map=True,saving_folder=saving_folder)
        jacobi_val_res += extra_res[0] * batch_size
        jacobi_num_res += extra_res[1] * batch_size
        average_jacobi_masked_res += extra_res[2]*batch_size
        records_jacobi_val_np[i] = extra_res[0]
        records_jacobi_num_np[i] = extra_res[1]
        average_jacobi_masked_np[i] = extra_res[2]
        print("id {} and current pair name is : {}".format(i,os.path.split(f_path)[-1]))
        print('the running averge jocobi val sum is {}'.format(jacobi_val_res/(i+1)/batch_size))
        print('the running averge jocobi num sum is {}'.format(jacobi_num_res/(i+1)/batch_size))
        print('the running averge jocobi masked average is {}'.format(average_jacobi_masked_res/(i+1)/batch_size))
    jacobi_val_res = jacobi_val_res/num_samples
    jacobi_num_res = jacobi_num_res/num_samples
    average_jacobi_masked_res = average_jacobi_masked_res/num_samples
    print("the average {}_ jacobi val sum: {}  :".format('test', jacobi_val_res))
    print("the average {}_ jacobi num sum: {}  :".format('test', jacobi_num_res))
    print("the average {}_ average_jacobi_masked average: {}  :".format('test', average_jacobi_masked_res))
    np.save(os.path.join(saving_folder,'records_jacobi'),records_jacobi_val_np)
    np.save(os.path.join(saving_folder,'records_jacobi_num'),records_jacobi_num_np)
    np.save(os.path.join(saving_folder,'records_average_jacobi_masked'),average_jacobi_masked_np)




deformation_mask_folder_path = "/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/reg/res/records/original_sz"
deformation_path_list =glob.glob(os.path.join(deformation_mask_folder_path,'*img_inv_phi.nii.gz'), recursive=True)
f_mask_list =[f.replace('img_inv_phi','img_moving_l') for f in deformation_path_list]
fname_list = [get_file_name(f).replace("_img_inv_phi","") for f in deformation_path_list]
saving_path = "/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/jacobi_analysis"
compute_jacobi(deformation_path_list,fname_list, saving_path, f_mask_list,itk_format=False)