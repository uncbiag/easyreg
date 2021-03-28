import SimpleITK as sitk
import numpy as np
import os
from easyreg.utils import resample_image
from mermaid.utils import compute_warped_image_multiNC
import torch
from easyreg.net_utils import gen_identity_map
from easyreg.demons_utils import sitk_grid_sampling
import tools.image_rescale as  ires

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
"""
the behavior of the itk is not clear, it would first move the source based on a displacement map with the moving size (here we assumed, maybe wrong should also check the map with target size) but apply on the target image
"""


moving_path ="/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_moving.nii.gz"
target_path ="/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_target.nii.gz"
disp_path ="/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_disp.h5"
inv_disp_path ="/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_inv_disp.h5"
mermaid_transform_path = "/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_phi.nii.gz"
mermaid_inv_transform_path = "/playpen-raid1/zyshen/data/demo_for_lung_reg/reg/res/records/original_sz/11769X_EXP_STD_BWH_COPD_img_11769X_INSP_STD_BWH_COPD_img_inv_phi.nii.gz"

moving_itk = sitk.ReadImage(moving_path)
target_itk = sitk.ReadImage(target_path)
trans_itk = sitk.ReadTransform(disp_path)
warped_itk = sitk_grid_sampling(target_itk,moving_itk, trans_itk)
inv_trans_itk = sitk.ReadTransform(inv_disp_path)
inv_warped_itk = sitk_grid_sampling(moving_itk,target_itk, inv_trans_itk)

moving_np = sitk.GetArrayFromImage(moving_itk).astype(np.float32)
target_np = sitk.GetArrayFromImage(target_itk).astype(np.float32)
mermaid_phi = sitk.GetArrayFromImage(sitk.ReadImage(mermaid_transform_path)).transpose(3, 2, 1,0)
mermaid_inv_phi = sitk.GetArrayFromImage(sitk.ReadImage(mermaid_inv_transform_path)).transpose(3, 2, 1,0)

phi_sz = np.array(mermaid_phi.shape)
spacing = 1./(np.array(phi_sz[1:])-1)
moving = torch.from_numpy(moving_np[None][None])
mermaid_phi = torch.from_numpy(mermaid_phi[None])
warped_mermaid = compute_warped_image_multiNC(moving, mermaid_phi, spacing, 1, zero_boundary=True)

inv_phi_sz = np.array(mermaid_inv_phi.shape)
spacing = 1./(np.array(inv_phi_sz[1:])-1)
target = torch.from_numpy(target_np[None][None])
mermaid_inv_phi = torch.from_numpy(mermaid_inv_phi[None])
inv_warped_mermaid = compute_warped_image_multiNC(target, mermaid_inv_phi, spacing, 1, zero_boundary=True)

output_path = "/playpen-raid1/zyshen/data/demo_for_lung_reg"
sitk.WriteImage(warped_itk, os.path.join(output_path,"warped_itk.nii.gz"))
ires.save_image_with_given_reference(warped_mermaid,reference_list=[target_path],path=output_path,fname=["warped_mermaid"])
sitk.WriteImage(inv_warped_itk, os.path.join(output_path,"inv_warped_itk.nii.gz"))
ires.save_image_with_given_reference(inv_warped_mermaid,reference_list=[moving_path],path=output_path,fname=["inv_warped_mermaid"])



