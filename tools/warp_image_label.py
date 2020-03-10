import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from easyreg.reg_data_utils import read_txt_into_list, get_file_name
from tools.image_rescale import save_image_with_given_reference
import SimpleITK as sitk
import torch
import numpy as np
from glob import glob
from mermaid.utils import compute_warped_image_multiNC, resample_image

def compute_warped_image_label(img_label_txt_pth,phi_pth,phi_type, saving_pth):
    img_label_pth_list = read_txt_into_list(img_label_txt_pth)
    phi_pth_list = glob(os.path.join(phi_pth,phi_type))
    f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
    img_list = [f(pth[0]) for pth in img_label_pth_list]
    label_list = [f(pth[1]) for pth in img_label_pth_list]
    num_img = len(img_list)
    for i in range(num_img):
        fname = get_file_name(img_label_pth_list[i][0])
        img = torch.Tensor(img_list[i][None][None])
        label = torch.Tensor(label_list[i][None][None])
        f_phi = lambda x: get_file_name(x).find(fname)==0
        phi_sub_list = list(filter(f_phi, phi_pth_list))
        num_aug = len(phi_sub_list)
        phi_list = [f(pth) for pth in phi_sub_list]
        img = img.repeat(num_aug,1,1,1,1)
        label = label.repeat(num_aug,1,1,1,1)
        phi = np.stack(phi_list,0)
        phi = np.transpose(phi,(0,4,3,2,1))
        phi = torch.Tensor(phi)
        sz = np.array(img.shape[2:])
        spacing = 1./(sz-1)
        phi, _ = resample_image(phi,spacing,[1,3]+list(img.shape[2:]))
        warped_img = compute_warped_image_multiNC(img,phi,spacing,spline_order=1,zero_boundary=True)
        warped_label = compute_warped_image_multiNC(label,phi,spacing,spline_order=0,zero_boundary=True)
        save_image_with_given_reference(warped_img,[img_label_pth_list[i][0]]*num_aug,saving_pth,[get_file_name(pth).replace("_phi","")+'_warped' for pth in phi_sub_list])
        save_image_with_given_reference(warped_label,[img_label_pth_list[i][0]]*num_aug,saving_pth,[get_file_name(pth).replace("_phi","")+'_label' for pth in phi_sub_list])


# img_label_txt_pth = "/playpen-raid/zyshen/data/lpba_seg_resize/test/file_path_list.txt"
# phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records"
# phi_type = '*_phi.nii.gz'
# saving_path = "/playpen-raid/zyshen/data/lpba_seg_resize/warped_img_label"
img_label_txt_pth = "/playpen-raid/zyshen/data/oai_seg/test/file_path_list.txt"
phi_pth = "/playpen-raid/zyshen/data/oai_reg/test_aug/reg/res/records"
phi_type = '*_phi.nii.gz'
saving_path = "/playpen-raid/zyshen/data/oai_seg/warped_img_label"
compute_warped_image_label(img_label_txt_pth,phi_pth,phi_type, saving_path)



