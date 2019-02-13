import SimpleITK as sitk
import os
import numpy as np
from data_pre.reg_data_utils import write_list_into_txt, get_file_name

def factor_tuple(input,factor):
    input_np = np.array(list(input))
    input_np = input_np*factor
    return tuple(list(input_np))

def __read_and_clean_itk_info(path):
    return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))

def resize_input_img_and_save_it_as_tmp(img_pth, is_label=False,fname=None,debug_path=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        img_org = sitk.ReadImage(img_pth)
        img = __read_and_clean_itk_info(img_pth)
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud([0.5,0.5,0.5])
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        fpth = os.path.join(debug_path,fname)
        img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth


def init_env(output_path, source_path_list, target_path_list, l_source_path_list=None, l_target_path_list=None):
    """
    :param task_full_path:  the path of a completed task
    :param source_path: path of the source image
    :param target_path: path of the target image
    :param l_source: path of the label of the source image
    :param l_target: path of the label of the target image
    :return: None
    """
    file_num = len(source_path_list)
    assert len(source_path_list) == len(target_path_list)
    if l_source_path_list is not None and l_target_path_list is not None:
        assert len(source_path_list) == len(l_source_path_list)
        file_list = [[source_path_list[i], target_path_list[i],l_source_path_list[i],l_target_path_list[i]] for i in range(file_num)]
    else:
        file_list = [[source_path_list[i], target_path_list[i]] for i in range(file_num)]
    os.makedirs(os.path.join(output_path,'reg/test'),exist_ok=True)
    os.makedirs(os.path.join(output_path,'reg/res'),exist_ok=True)
    pair_txt_path =  os.path.join(output_path,'reg/test/pair_path_list.txt')
    fn_txt_path =   os.path.join(output_path,'reg/test/pair_name_list.txt')
    fname_list = [get_file_name(file_list[i][0])+'_'+get_file_name(file_list[i][1]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path,file_list)
    write_list_into_txt(fn_txt_path,fname_list)
    root_path = output_path
    data_task_name = 'reg'
    cur_task_name = 'res'
    return root_path

def loading_img_list_from_files(path):
    from data_pre.reg_data_utils import read_txt_into_list
    path_list = read_txt_into_list(path)
    num_pair = len(path_list)
    assert len(path_list[0])>=2
    has_label = True if len(path_list[0])==4 else False
    source_path_list = [path_list[i][0] for i in range(num_pair)]
    target_path_list = [path_list[i][1] for i in range(num_pair)]
    l_source_path_list = None
    l_target_path_list = None
    if has_label:
        l_source_path_list = [path_list[i][2] for i in range(num_pair)]
        l_target_path_list = [path_list[i][3] for i in range(num_pair)]
    return source_path_list, target_path_list, l_source_path_list, l_target_path_list




debug_path = '/playpen/zyshen/data_pre/down_sampled_training_for_intra/'

img_list_txt_path = '/playpen/zyshen/debugs/get_val_and_debug_res/debug.txt'
output_path = '/playpen/zyshen/debugs/zhengyang'
source_path_list, target_path_list, l_source_path_list, l_target_path_list = loading_img_list_from_files(
    img_list_txt_path)
init_env(output_path,source_path_list,target_path_list,l_source_path_list,l_target_path_list)
