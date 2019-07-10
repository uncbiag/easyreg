import SimpleITK as sitk
from data_pre.reg_data_utils import write_list_into_txt, get_file_name
from model_pool.utils import *
import mermaid.image_sampling as py_is
import mermaid.utils as py_utils
from mermaid.data_wrapper import MyTensor

def factor_tuple(input,factor):
    input_np = np.array(list(input))
    input_np = input_np*factor
    return tuple(list(input_np))

def __read_and_clean_itk_info(input):
    if isinstance(input,str):
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(input)))
    else:
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(input))




def resize_input_img_and_save_it_as_tmp(img_input,resize_factor=(0.5,0.5,0.5), is_label=False,keep_physical=True,fname=None,saving_path=None,fixed_sz=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :param  fix sz should be refered to numpy coord
        :return:
        """
        if isinstance(img_input, str):
            img_org = sitk.ReadImage(img_input)
            img = __read_and_clean_itk_info(img_input)
        else:
            img_org = img_input
            img =__read_and_clean_itk_info(img_input)
        # resampler= sitk.ResampleImageFilter()
        # resampler.SetSize(img.GetSize())
        # bspline = sitk.BSplineTransformInitializer(img, (5, 5, 5), 2)
        # resampler.SetTransform(bspline)
        # img = resampler.Execute(img)
        dimension =3

        img_sz = img.GetSize()
        if not fixed_sz:
            factor = np.flipud(resize_factor)
        else:
            fixed_sz  = np.flipud(fixed_sz)
            factor = [fixed_sz[i]/img_sz[i] for i in range(len(img_sz))]
        resize = not all([factor == 1 for factor in resize_factor])
        if resize:
            resampler = sitk.ResampleImageFilter()
            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
            after_size = [int(sz) for sz in after_size]
            if fixed_sz is not None:
                for i in range(len(fixed_sz)):
                    assert fixed_sz[i]==after_size[i]
            matrix[0, 0] = 1. / factor[0]
            matrix[1, 1] = 1. / factor[1]
            matrix[2, 2] = 1. / factor[2]
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
        if fname is not None:
            os.makedirs(saving_path, exist_ok=True)
            fpth = os.path.join(saving_path, fname)
        else:
            os.makedirs(os.path.split(saving_path)[0], exist_ok=True)
            fpth = saving_path
        if keep_physical:
            img_resampled.SetSpacing(resize_spacing(img_sz, img_org.GetSpacing(), factor))
            img_resampled.SetOrigin(img_org.GetOrigin())
            img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth


def resample_warped_phi_and_image(source,phi,spacing, new_sz,using_file_list=True):
    if using_file_list:
        num_s = len(source)
        s = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in source]
        sz = [num_s,1]+list(s[0].shape)
        source = np.stack(s,axis=0)
        source = source.reshape(*sz)
        source = MyTensor(source)


    new_phi,new_spacing = resample_image( phi, spacing, new_sz, 1, zero_boundary=True)
    warped = py_utils.compute_warped_image_multiNC(source, new_phi, new_spacing, 1, zero_boundary=True)
    return new_phi, warped, new_spacing


def save_transfrom(transform,path=None, fname=None,using_affine=False):

    if not using_affine:
        if type(transform) == torch.Tensor:
            transform = transform.detach().cpu().numpy()
        import nibabel as nib
        for i in range(transform.shape[0]):
            phi = nib.Nifti1Image(transform[i], np.eye(4))
            fn = '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
            nib.save(phi, os.path.join(path, fn+'_phi.nii.gz'))
    else:
        affine_param = transform
        if isinstance(affine_param, list):
            affine_param =affine_param[0]
        affine_param = affine_param.detach().cpu().numpy()
        for i in range(affine_param.shape[0]):
            fn =  '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
            np.save(os.path.join(path, fn + '_affine.npy'), affine_param[i])


def save_image_with_given_reference(img=None,reference_list=None,path=None,fname=None):

    num_img = len(fname)
    os.makedirs(path,exist_ok=True)
    for i in range(num_img):
        img_ref = sitk.ReadImage(reference_list[i])
        if img is not None:
            if type(img) == torch.Tensor:
                img = img.detach().cpu().numpy()
            spacing_ref = img_ref.GetSpacing()
            direc_ref = img_ref.GetDirection()
            orig_ref = img_ref.GetDirection()
            img_itk = sitk.GetImageFromArray(img[i,0])
            img_itk.SetSpacing(spacing_ref)
            img_itk.SetDirection(direc_ref)
            img_itk.SetOrigin(orig_ref)
        else:
            img_itk=img_ref
        fn =  '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
        fpath = os.path.join(path,fn+'.nii.gz')
        sitk.WriteImage(img_itk,fpath)








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




# debug_path = '/playpen/zyshen/data_pre/down_sampled_training_for_intra/'
#
# img_list_txt_path = '/playpen/zyshen/debugs/get_val_and_debug_res/debug.txt'
# output_path = '/playpen/zyshen/debugs/zhengyang'
# source_path_list, target_path_list, l_source_path_list, l_target_path_list = loading_img_list_from_files(
#     img_list_txt_path)
# init_env(output_path,source_path_list,target_path_list,l_source_path_list,l_target_path_list)
# path = '/playpen/zyshen/debugs/dct/OAS30006_MR_d0166_brain_origin.nii.gz'
# saving_path='/playpen/zyshen/debugs/dct'
# resize_input_img_and_save_it_as_tmp(path,is_label=False,saving_path=saving_path,fname='bspline_test.nii.gz')
