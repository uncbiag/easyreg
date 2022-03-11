import os
import numpy as np
import torch
import SimpleITK as sitk
from tools.module_parameters import ParameterDict
from easyreg.mermaid_net import MermaidNet as model
from easyreg.utils import resample_image
import mermaid.utils as py_utils


def resize_img(img, img_after_resize=None, is_mask=False):
    """
    :param img: sitk input, factor is the outputs_ize/patched_sized
    :param img_after_resize: list, img_after_resize, image size after resize in itk coordinate
    :return:
    """
    img_sz = img.GetSize()
    if img_after_resize is not None:
        img_after_resize = img_after_resize
    else:
        img_after_resize = img_sz
    resize_factor = np.array(img_after_resize) / np.flipud(img_sz)
    spacing_factor = (np.array(img_after_resize) - 1) / (np.flipud(img_sz) - 1)
    resize = not all([factor == 1 for factor in resize_factor])
    if resize:
        resampler = sitk.ResampleImageFilter()
        dimension = 3
        factor = np.flipud(resize_factor)
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] = 1. / spacing_factor[0]
        matrix[1, 1] = 1. / spacing_factor[1]
        matrix[2, 2] = 1. / spacing_factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
        img_resampled = resampler.Execute(img)
    else:
        img_resampled = img
    return img_resampled, resize_factor



def convert_itk_to_support_deepnet(img_sitk, is_mask=False,device=torch.device("cuda:0")):
    img_sz_after_resize = [80,192,192]
    img_sitk = sitk.GetImageFromArray(sitk.GetArrayFromImage(img_sitk))
    img_after_resize,_ = resize_img(img_sitk,img_sz_after_resize, is_mask=is_mask)
    img_numpy = sitk.GetArrayFromImage(img_after_resize)
    if not is_mask:
        img_numpy = img_numpy*2-1
    return torch.Tensor(img_numpy.astype(np.float32))[None][None].to(device)



def convert_output_into_itk_support_format(source_itk,target_itk, l_source_itk, l_target_itk, phi,spacing):
    phi = (phi+1)/2 # here we assume the phi take the [-1,1] coordinate, usually used by deep network
    new_phi = None
    warped = None
    l_warped = None
    new_spacing = None
    if source_itk is not None:
        s =  sitk.GetArrayFromImage(source_itk)
        t =  sitk.GetArrayFromImage(target_itk)
        sz_t = [1, 1] + list(t.shape)
        source = torch.from_numpy(s[None][None]).to(phi.device)
        new_phi, new_spacing = resample_image(phi, spacing, sz_t, 1, zero_boundary=True)
        warped = py_utils.compute_warped_image_multiNC(source, new_phi, new_spacing, 1, zero_boundary=True)

    if l_source_itk is not None:
        ls = sitk.GetArrayFromImage(l_source_itk).astype(np.float32)
        lt = sitk.GetArrayFromImage(l_target_itk).astype(np.float32)
        sz_lt = [1, 1] + list(lt.shape)
        l_source = torch.from_numpy(ls[None][None]).to(phi.device)
        if new_phi is None:
            new_phi, new_spacing = resample_image(phi, spacing, sz_lt, 1, zero_boundary=True)
        l_warped = py_utils.compute_warped_image_multiNC(l_source, new_phi, new_spacing, 0, zero_boundary=True)
    return new_phi, warped, l_warped


def predict(source_itk, target_itk,source_mask_itk=None, target_mask_itk=None,setting_path="", model_path="",device=torch.device("cuda:0")):
    opt = ParameterDict()
    opt.load_JSON(setting_path)
    source = convert_itk_to_support_deepnet(source_itk,device=device)
    target = convert_itk_to_support_deepnet(target_itk, device=device)
    source_mask = convert_itk_to_support_deepnet(source_mask_itk,is_mask=True) if source_mask_itk is not None else None
    target_mask = convert_itk_to_support_deepnet(target_mask_itk,is_mask=True) if target_mask_itk is not None else None
    network = model(img_sz=[80,192,192],opt=opt)
    network.load_pretrained_model(model_path)
    network.to(device)
    network.train(False)
    with torch.no_grad():
        warped, composed_map, _ = network.forward(source, target, source_mask, target_mask)
        composed_inv_map = network.get_inverse_map(use_01=False)
    spacing = 1./(np.array(warped.shape[2:])-1)
    del network
    full_inv_composed_map, full_inv_warped,l_full_inv_warped = convert_output_into_itk_support_format(target_itk,source_itk, target_mask_itk, source_mask_itk, composed_inv_map,spacing)
    full_inv_composed_map = full_inv_composed_map.detach().cpu().squeeze().numpy()
    # save_3D_img_from_itk(source_itk, "/playpen-raid1/zyshen/debug/debug_oai_model_source_itk.nii.gz")
    # save_3D_img_from_itk(target_itk, "/playpen-raid1/zyshen/debug/debug_oai_model_target_itk.nii.gz")
    # # save_3D_img_from_numpy(full_warped.squeeze().cpu().numpy(),
    # #                        "/playpen-raid1/zyshen/debug/debug_lin_model_st.nii.gz",
    # #                        target_itk.GetSpacing(), target_itk.GetOrigin(), target_itk.GetDirection())
    # save_3D_img_from_numpy(full_inv_warped.squeeze().cpu().numpy(),"/playpen-raid1/zyshen/debug/debug_oai_model_ts.nii.gz",
    #                        source_itk.GetSpacing(), source_itk.GetOrigin(), source_itk.GetDirection())
    return full_inv_composed_map



def get_file_name(img_path):
    get_fn = lambda x: os.path.split(x)[-1]
    file_name = get_fn(img_path).split(".")[0]
    return file_name

if __name__ == "__main__":
    """
    input:
    here we assume the input have been preprocessed (normalized into [0,1])
    mesh_path(in source coordinate)
    source path
    target path
    source_mask_path(optional)
    target_mask_path(optional)
    
    output: deformed mesh(in target coordinate)
    """
    source_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image.nii.gz"
    target_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image.nii.gz"
    source_mask_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_label_all.nii.gz"
    target_mask_path = "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_label_all.nii.gz"
    setting_path = "./demo_settings/mermaid/eval_network_vsvf/cur_task_setting.json"
    model_path = "./demo_saved_models/mermaid/eval_network_vsvf/model"
    source_itk = sitk.ReadImage(source_path)
    target_itk = sitk.ReadImage(target_path)
    source_mask_itk = sitk.ReadImage(source_mask_path) if len(source_mask_path) else None
    target_mask_itk = sitk.ReadImage(target_mask_path) if len(target_mask_path) else None
    full_inv_composed_map = predict(source_itk, target_itk,source_mask_itk,target_mask_itk,setting_path, model_path=model_path)
