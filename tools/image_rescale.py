import SimpleITK as sitk
from easyreg.reg_data_utils import write_list_into_txt, generate_pair_name
from easyreg.utils import *
import mermaid.utils as py_utils
from mermaid.data_wrapper import MyTensor



def __read_and_clean_itk_info(input):
    if isinstance(input,str):
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(input)))
    else:
        return sitk.GetImageFromArray(sitk.GetArrayFromImage(input))




def resize_input_img_and_save_it_as_tmp(img_input,resize_factor=(1.0,1.0,1.0), is_label=False,keep_physical=True,fname=None,saving_path=None,fixed_sz=None):
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
            after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
            spacing_factor = [(after_size[i]-1)/(img_sz[i]-1) for i in range(len(img_sz))]
        else:
            fixed_sz  = np.flipud(fixed_sz)
            factor = [fixed_sz[i]/img_sz[i] for i in range(len(img_sz))]
            spacing_factor = [(fixed_sz[i]-1)/(img_sz[i]-1) for i in range(len(img_sz))]
        resize = not all([f == 1 for f in factor])
        if resize:
            resampler = sitk.ResampleImageFilter()
            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
            after_size = [int(sz) for sz in after_size]
            if fixed_sz is not None:
                for i in range(len(fixed_sz)):
                    assert fixed_sz[i]==after_size[i]
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





def resample_warped_phi_and_image(source_path,target_path, l_source_path, l_target_path, phi,spacing):
    new_phi = None
    warped = None
    l_warped = None
    new_spacing = None
    if source_path is not None:
        s = sitk.GetArrayFromImage(sitk.ReadImage(source_path)).astype(np.float32)
        t = sitk.GetArrayFromImage(sitk.ReadImage(target_path)).astype(np.float32)
        sz_t = [1, 1] + list(t.shape)
        source = torch.from_numpy(s[None][None]).to(phi.device)
        new_phi, new_spacing = resample_image(phi, spacing, sz_t, 1, zero_boundary=True)
        warped = py_utils.compute_warped_image_multiNC(source, new_phi, new_spacing, 1, zero_boundary=True)


    if l_source_path is not None:
        ls = sitk.GetArrayFromImage(sitk.ReadImage(l_source_path)).astype(np.float32)
        lt = sitk.GetArrayFromImage(sitk.ReadImage(l_target_path)).astype(np.float32)
        sz_lt = [1, 1] + list(lt.shape)
        l_source = torch.from_numpy(ls[None][None]).to(phi.device)
        if new_phi is None:
            new_phi, new_spacing = resample_image(phi, spacing, sz_lt, 1, zero_boundary=True)
        l_warped = py_utils.compute_warped_image_multiNC(l_source, new_phi, new_spacing, 0, zero_boundary=True)

    return new_phi, warped,l_warped, new_spacing


#
# def resample_warped_phi_and_image(source_path,target_path, l_source_path, l_target_path, phi,spacing):
#     new_phi = None
#     warped = None
#     l_warped = None
#     new_spacing = None
#     if source_path is not None:
#         s = sitk.GetArrayFromImage(sitk.ReadImage(source_path)).astype(np.float32)
#         #t = sitk.GetArrayFromImage(sitk.ReadImage(target_path)).astype(np.float32)
#         sz_t = [1, 1] + list(s.shape)
#         source = torch.from_numpy(s[None][None]).to(phi.device)
#         new_phi, new_spacing = resample_image(phi, spacing, sz_t, 1, zero_boundary=True)
#         warped = py_utils.compute_warped_image_multiNC(source, new_phi, new_spacing, 1, zero_boundary=True)
#
#
#     if l_source_path is not None:
#         ls = sitk.GetArrayFromImage(sitk.ReadImage(l_source_path)).astype(np.float32)
#         #lt = sitk.GetArrayFromImage(sitk.ReadImage(l_target_path)).astype(np.float32)
#         sz_lt = [1, 1] + list(ls.shape)
#         l_source = torch.from_numpy(ls[None][None]).to(phi.device)
#         if new_phi is None:
#             new_phi, new_spacing = resample_image(phi, spacing, sz_lt, 1, zero_boundary=True)
#         l_warped = py_utils.compute_warped_image_multiNC(l_source, new_phi, new_spacing, 0, zero_boundary=True)
#
#     return new_phi, warped,l_warped, new_spacing



def save_transform_with_reference(transform, spacing,moving_reference_list, target_reference_list, path=None, fname_list=None,save_disp_into_itk_format=True):
    if not save_disp_into_itk_format:
        save_transfrom(transform, spacing, path, fname_list)
    else:
        save_transform_itk(transform,spacing,moving_reference_list, target_reference_list, path, fname_list)




def save_transform_itk(transform,spacing,moving_list,target_list, path, fname_list):
    from mermaid.utils import identity_map

    if type(transform) == torch.Tensor:
        transform = transform.detach().cpu().numpy()


    for i in range(transform.shape[0]):
        cur_trans = transform[i]
        img_sz = np.array(transform.shape[2:])

        moving_ref = sitk.ReadImage(moving_list[i])
        moving_spacing_ref = moving_ref.GetSpacing()
        moving_direc_ref = moving_ref.GetDirection()
        moving_orig_ref = moving_ref.GetOrigin()
        target_ref = sitk.ReadImage(target_list[i])
        target_spacing_ref = target_ref.GetSpacing()
        target_direc_ref = target_ref.GetDirection()
        target_orig_ref = target_ref.GetOrigin()

        id_np_moving = identity_map(img_sz, np.flipud(moving_spacing_ref))
        id_np_target = identity_map(img_sz, np.flipud(target_spacing_ref))
        factor = np.flipud(moving_spacing_ref) / spacing
        factor  = factor.reshape(3,1,1,1)

        moving_direc_matrix = np.array(moving_direc_ref).reshape(3, 3)
        target_direc_matrix = np.array(target_direc_ref).reshape(3, 3)
        cur_trans = np.matmul(moving_direc_matrix, permute_trans(id_np_moving + cur_trans * factor).reshape(3, -1)) \
                    - np.matmul(target_direc_matrix, permute_trans(id_np_target).reshape(3, -1))
        cur_trans = cur_trans.reshape(id_np_moving.shape)
        fn = '{}_batch_'.format(i) + fname_list if not type(fname_list) == list else fname_list[i]
        saving_path =  os.path.join(path, fn + '.h5')

        bias = np.array(target_orig_ref)-np.array(moving_orig_ref)
        bias = -bias.reshape(3,1,1,1)
        transform_physic = cur_trans +bias

        trans = get_transform_with_itk_format(transform_physic,target_spacing_ref, target_orig_ref,target_direc_ref)
        #sitk.WriteTransform(trans, saving_path)
        # Retrive the DField from the Transform
        dfield = trans.GetDisplacementField()
        # Fitting a BSpline from the Deformation Field
        bstx = dfield2bspline(dfield, verbose=True)

        # Save the BSpline Transform
        sitk.WriteTransform(bstx, saving_path.replace('.h5', '.tfm'))

def permute_trans(trans):
    trans_new = np.zeros_like(trans)
    trans_new[0,...] = trans[2,...]
    trans_new[1,...] = trans[1,...]
    trans_new[2,...] = trans[0,...]
    return trans_new


def save_transfrom(transform,spacing, path=None, fname=None,using_affine=False):
    if not using_affine:
        if type(transform) == torch.Tensor:
            transform = transform.detach().cpu().numpy()
        img_sz = np.array(transform.shape[2:])
        # mapping into 0, 1 coordinate
        for i in range(len(img_sz)):
            transform[:, i, ...] = transform[:, i, ...] / ((img_sz[i] - 1) * spacing[i])
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
        for i in range(affine_param.shape[0]): # todo the bias part of the affine param should also normalized into non-physical space [0,1]
            fn =  '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
            np.save(os.path.join(path, fn + '_affine.npy'), affine_param[i])


def save_image_with_given_reference(img=None,reference_list=None,path=None,fname=None):
    """

    :param img: Nx1xDxHxW
    :param reference_list: N list
    :param path: N list
    :param fname: N list
    :return:
    """

    num_img = len(fname) if fname is not None else 0
    os.makedirs(path,exist_ok=True)
    for i in range(num_img):
        img_ref = sitk.ReadImage(reference_list[i])
        if img is not None:
            if type(img) == torch.Tensor:
                img = img.detach().cpu().numpy()
            spacing_ref = img_ref.GetSpacing()
            direc_ref = img_ref.GetDirection()
            orig_ref = img_ref.GetOrigin()
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
    fname_list = [generate_pair_name([file_list[i][0],file_list[i][1]]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path,file_list)
    write_list_into_txt(fn_txt_path,fname_list)
    root_path = output_path
    data_task_name = 'reg'
    cur_task_name = 'res'
    return root_path



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
