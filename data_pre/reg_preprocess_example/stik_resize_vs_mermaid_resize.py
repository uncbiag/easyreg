import numpy as np
import SimpleITK as sitk
from easyreg.utils import get_resampled_image
import torch
def resize_img(img, is_label=False,img_after_resize=None):
    """
    :param img: sitk input, factor is the outputs_ize/patched_sized
    :return:
    """
    img_sz = img.GetSize()
    if img_after_resize is None:
        img_after_resize = np.flipud(img_sz)
    resize_factor = np.array(img_after_resize) / np.flipud(img_sz)
    spacing_factor = (np.array(img_after_resize)-1) / (np.flipud(img_sz)-1)
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
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
    else:
        img_resampled = img
    return img_resampled, resize_factor


def read_and_clean_itk_info(path):
    if path is not None:
        img = sitk.ReadImage(path)
        spacing_sitk = img.GetSpacing()
        img_sz_sitk = img.GetSize()
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(img)), np.flipud(spacing_sitk), np.flipud(img_sz_sitk)
    else:
        return None, None, None



img_path = "/playpen-raid2/Data/Lung_Registration_clamp_normal_transposed/10056H/10056H_EXP_STD_NJC_COPD_img.nii.gz"
img_sitk, original_spacing, original_sz = read_and_clean_itk_info(img_path)
sampled_size = np.array([160,160,160])
img_np = sitk.GetArrayFromImage(img_sitk)
resized_img, resize_factor = resize_img(img_sitk,img_after_resize=sampled_size)
img_sitkresized_np = sitk.GetArrayFromImage(resized_img)
print(img_np.shape)
spacing = np.array([1,1,1])
img_merresized_img = get_resampled_image(torch.Tensor(img_np)[None][None], spacing,np.array([1,1,160,160,160]), 1, zero_boundary=False)
img_merresized_img = img_merresized_img.numpy().squeeze()
print((img_sitkresized_np- img_merresized_img).sum())
img_sitk = sitk.GetImageFromArray(img_sitkresized_np.astype(np.float32))
sitk.WriteImage(img_sitk,"/playpen-raid1/zyshen/debug/img_sitkresized_np.nii.gz")
img_sitk = sitk.GetImageFromArray(img_merresized_img.astype(np.float32))
sitk.WriteImage(img_sitk,"/playpen-raid1/zyshen/debug/img_merresized_img.nii.gz")


