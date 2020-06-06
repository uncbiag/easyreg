import SimpleITK as sitk
import numpy
from mermaid.utils import *
import torch
import tools.image_rescale as  ires
from easyreg.net_utils import gen_identity_map
from easyreg.demons_utils import sitk_grid_sampling


# img_org_path = "/playpen-raid1/zyshen/debug/9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image.nii.gz"
# img_tar_path = "/playpen-raid1/zyshen/debug/9403165_20060316_SAG_3D_DESS_LEFT_016610900302_image.nii.gz"
# moving_path = "/playpen-raid1/zyshen/debug/9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image_cleaned.nii.gz"
# target_path = "/playpen-raid1/zyshen/debug/9403165_20060316_SAG_3D_DESS_LEFT_016610900302_image_cleaned.nii.gz"
#
# moving_org = sitk.ReadImage(img_org_path)
# spacing_ref = moving_org.GetSpacing()
# direc_ref = moving_org.GetDirection()
# orig_ref = moving_org.GetOrigin()
# img_itk = sitk.GetImageFromArray(sitk.GetArrayFromImage(moving_org))
# img_itk.SetSpacing(spacing_ref)
# img_itk.SetDirection(direc_ref)
# img_itk.SetOrigin(orig_ref)
# sitk.WriteImage(img_itk,moving_path)
#
#
# target_org = sitk.ReadImage(img_tar_path)
# spacing_ref = target_org.GetSpacing()
# spacing_ref = tuple(s*2 for s in spacing_ref)
# direc_ref = target_org.GetDirection()
# orig_ref = target_org.GetOrigin()
# img_itk = sitk.GetImageFromArray(sitk.GetArrayFromImage(target_org))
# img_itk.SetSpacing(spacing_ref)
# img_itk.SetDirection(direc_ref)
# img_itk.SetOrigin(orig_ref)
# sitk.WriteImage(img_itk,target_path)
#
#
# moving = sitk.ReadImage(moving_path)
# target = sitk.ReadImage(target_path)
# moving_np  = sitk.GetArrayFromImage(moving)
#
#
# img_sz = np.array(moving_np.shape)
# spacing = 1./(np.array(img_sz)-1)
#
# id_np= gen_identity_map(img_sz, resize_factor=1., normalized=True)
# id_np = (id_np+1.)/2
# #disp_np = np.zeros([3]+list(moving_np.shape)).astype(np.float32)
# disp_np = np.random.rand(3,80,192,192).astype(np.float32)/20
# disp_np[0] = disp_np[0]+0.03
# disp_np[1] = disp_np[1]+0.05
# disp_np[2] = disp_np[2]+0.09
# phi_np = id_np + disp_np
#
# phi = torch.Tensor(phi_np)
# warped_mermaid = compute_warped_image_multiNC(torch.Tensor(moving_np)[None][None],phi[None],spacing,spline_order=1,zero_boundary=True)
# ires.save_image_with_given_reference(warped_mermaid,[target_path],"/playpen-raid1/zyshen/debug",["9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image_warped"])
#
# trans  =ires.save_transform_itk(disp_np[None], spacing,[moving_path],[target_path],"/playpen-raid1/zyshen/debug",["9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image"] )
# warped_itk = sitk_grid_sampling(target, moving, trans)
# sitk_warped_path = "/playpen-raid1/zyshen/debug/9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image_warped_sitk.nii.gz"
# sitk.WriteImage(warped_itk, sitk_warped_path)
# print("Done")


############### reconstruct the lung  ##############33

moving_path ="/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/reg/res/records/original_sz/13074Y_EXP_STD_TEM_COPD_img_13074Y_INSP_STD_TEM_COPD_img_moving.nii.gz"
target_path ="/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/reg/res/records/original_sz/13074Y_EXP_STD_TEM_COPD_img_13074Y_INSP_STD_TEM_COPD_img_target.nii.gz"
disp_path ="/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/reg/res/records/original_sz/13074Y_EXP_STD_TEM_COPD_img_13074Y_INSP_STD_TEM_COPD_img_disp.h5"
inv_disp_path ="/playpen-raid1/zyshen/data/reg_new_lung/testing_lddmm/reg/res/records/original_sz/13074Y_EXP_STD_TEM_COPD_img_13074Y_INSP_STD_TEM_COPD_img_inv_disp.h5"
#trans = sitk.ReadTransform(disp_path)
inv_trans = sitk.ReadTransform(inv_disp_path)
#warped_itk = sitk_grid_sampling(sitk.ReadImage(target_path),sitk.ReadImage(moving_path), trans)
inv_warped_itk = sitk_grid_sampling(sitk.ReadImage(moving_path),sitk.ReadImage(target_path), inv_trans)
#sitk.WriteImage(warped_itk,"/playpen-raid1/zyshen/debug/warped.nii.gz")
sitk.WriteImage(inv_warped_itk,"/playpen-raid1/zyshen/debug/inv_warped.nii.gz")


