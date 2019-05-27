import os
import nibabel as nib
import SimpleITK as sitk
import numpy
import torch
from tools.visualize_registration_results import show_current_images


def read_img(path,is_sitk=True,is_viz=False):
    if is_sitk:
        img = sitk.ReadImage(path)
        img = torch.from_numpy(sitk.GetArrayFromImage(img))
        if not is_viz:
            img = img.view([1,1]+list(img.shape))
        else:
            img = img.view([1]+list(img.shape))
    else:
        img = nib.load(path)
        data = img.get_fdata()
        img =torch.from_numpy(data)
        img  = img.view([1]+list(img.shape))
    return img

def get_visual_param(save_path, pair_name):
    visual_param = {}
    visual_param['visualize'] = False
    visual_param['save_fig'] = True
    visual_param['save_fig_path'] = save_path
    visual_param['save_fig_path_byname'] = os.path.join(save_path, 'byname')
    visual_param['save_fig_path_byiter'] = os.path.join(save_path, 'byiter')
    visual_param['save_fig_num'] = 1
    visual_param['pair_path'] = [pair_name]
    visual_param['iter'] = '0'
    return visual_param


root_path = '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/rdmm_iter_wkw_formul_025_1_omt_2step_200sym_minstd_005_allinterp_maskv_epdffix/reg/res/records'
saved_path = '/playpen/zyshen/debugs/rdmm/iter_006/res'
os.makedirs(saved_path,exist_ok=True)

img_root_path = os.path.join(root_path,'3D')
extra_root_path = root_path
pair_name_list = ['9047800_20060306_SAG_3D_DESS_LEFT_016610874403_image_9905156_20050928_SAG_3D_DESS_LEFT_016610601603_image',
                  '9054866_20040920_SAG_3D_DESS_RIGHT_016610242012_image_9967358_20050105_SAG_3D_DESS_RIGHT_016610319712_image',
                  '9056363_20051010_SAG_3D_DESS_LEFT_016610100103_image_9803694_20041013_SAG_3D_DESS_LEFT_016610266313_image',
                  '9085290_20040915_SAG_3D_DESS_LEFT_016610243503_image_9663614_20050923_SAG_3D_DESS_LEFT_016610593102_image',
                  '9967358_20050105_SAG_3D_DESS_RIGHT_016610319712_image_9054866_20040920_SAG_3D_DESS_RIGHT_016610242012_image',
                  '9415074_20050726_SAG_3D_DESS_RIGHT_016610438114_image_9482482_20040929_SAG_3D_DESS_RIGHT_016610256112_image']

for pair_name in pair_name_list:
    s_pth = os.path.join(img_root_path,pair_name+'_test_iter_0_moving.nii.gz')
    t_pth = os.path.join(img_root_path,pair_name+'_test_iter_0_target.nii.gz')
    w_pth = os.path.join(img_root_path,pair_name+'_test_iter_0_warped.nii.gz')
    phi_pth =os.path.join(extra_root_path,pair_name+'_phi.nii.gz')
    jaco_pth =os.path.join(extra_root_path,pair_name+'_0000jacobi_img.nii')
    sw_pth =os.path.join(extra_root_path,pair_name+'_0000smoother_weight.nii.gz')
    #sw_pth =os.path.join(extra_root_path,pair_name+'_0000Inital_weight.nii.gz')
    s = read_img(s_pth)
    s =(s+1)/2
    t = read_img(t_pth)
    t =(t+1)/2
    w = read_img(w_pth)
    t =(t+1)/2
    phi = read_img(phi_pth,is_sitk=False)
    jaco = read_img(jaco_pth,is_viz=True)
    sw = read_img(sw_pth)
    visual_param=get_visual_param(saved_path,pair_name)
    show_current_images(0, iS=s,iT=t,iW=w,
                            iSL=s,iTL=s, iWL=s,
                            vizImages=jaco, vizName='jacobi_det',phiWarped=phi,
                            visual_param=visual_param,extraImages=sw, extraName= 'inital_w')