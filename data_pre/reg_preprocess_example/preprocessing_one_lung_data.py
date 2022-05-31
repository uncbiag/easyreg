import os
import numpy as np
import SimpleITK as sitk
from data_pre.reg_preprocess_example.img_sampler import DataProcessing


def process_image(img, is_seg=False):
    """
    :param img: numpy image
    :return:
    """
    if not is_seg:
        img[img<-1000] = -1000
        img[img>-200] = -200
        # img = normalize_intensity(img)
    else:
        img[img > 400] = 0
        img[img != 0] = 1
    return img



def process_high_to_raul_format(image_path,seg_path, saving_folder, output_spacing=(1.,1.,1.), output_size=(350,350,350)):
    """
    :param image_path: an image path
    :param seg_path:  a segmentation path
    :param saving_folder:  output saving folder
    :param output_spacing: output image/seg spacing
    :param output_size: output image/seg size
    :return:
    """

    img_sitk = sitk.ReadImage(image_path)
    seg_sitk = sitk.ReadImage(seg_path)
    processed_img = DataProcessing.resample_image_itk_by_spacing_and_size(img_sitk, output_spacing =np.array(output_spacing), output_size=output_size, output_type=None,
                                               interpolator=sitk.sitkBSpline, padding_value=-1000, center_padding=True)

    get_fname = lambda x: os.path.split(x)[1].split('.')[0]
    fname = get_fname(image_path)
    saving_path = os.path.join(saving_folder,fname + "_img.nii.gz")
    img_np = sitk.GetArrayFromImage(processed_img)
    img_np = process_image(img_np.astype(np.float32))
    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_img.SetOrigin(processed_img.GetOrigin())
    sitk_img.SetSpacing(processed_img.GetSpacing())
    sitk_img.SetDirection(processed_img.GetDirection())
    sitk.WriteImage(sitk_img,saving_path)
    processed_seg = DataProcessing.resample_image_itk_by_spacing_and_size(seg_sitk,
                                                                          output_spacing=np.array(output_spacing),
                                                                          output_size=output_size, output_type=None,
                                                                          interpolator=sitk.sitkNearestNeighbor,
                                                                          padding_value=0, center_padding=True)
    saving_path = os.path.join(saving_folder, fname + "_seg.nii.gz")
    seg_np = sitk.GetArrayFromImage(processed_seg)
    seg_np = process_image(seg_np.astype(np.float32),is_seg=True)
    sitk_seg = sitk.GetImageFromArray(seg_np)
    sitk_seg.SetOrigin(processed_seg.GetOrigin())
    sitk_seg.SetSpacing(processed_seg.GetSpacing())
    sitk_seg.SetDirection(processed_seg.GetDirection())
    sitk.WriteImage(sitk_seg, saving_path)
