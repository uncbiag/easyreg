import os
import numpy as np
import torch
import SimpleITK as sitk
import pyvista as pv
from easyreg.net_utils import gen_identity_map
from tools.image_rescale import permute_trans
from tools.module_parameters import ParameterDict
from easyreg.lin_unpublic_net import model
from easyreg.utils import resample_image, get_transform_with_itk_format, dfield2bspline
from tools.visual_tools import save_3D_img_from_numpy
import mermaid.utils as py_utils


dirlab_landmarks_folder = "/playpen-raid1/zyshen/lung_reg/evaluate/dirlab_landmarks"

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
def resample_image_itk_by_spacing_and_size(
        image,
        output_spacing,
        output_size,
        output_type=None,
        interpolator=sitk.sitkBSpline,
        padding_value=-1024,
        center_padding=True,
):
    """
    Image resampling using ITK
    :param image: simpleITK image
    :param output_spacing: numpy array or tuple. Output spacing
    :param output_size: numpy array or tuple. Output size
    :param output_type: simpleITK output data type. If None, use the same as 'image'
    :param interpolator: simpleITK interpolator (default: BSpline)
    :param padding_value: pixel padding value when a transformed pixel is outside of the image
    :return: tuple with simpleITK image and array with the resulting output spacing
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(output_size)
    resampler.SetDefaultPixelValue(padding_value)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(np.array(output_spacing))
    resampler.SetOutputPixelType(
        output_type if output_type is not None else image.GetPixelIDValue()
    )
    factor = np.asarray(image.GetSpacing()) / np.asarray(output_spacing).astype(
        np.float32
    )

    # Get new output origin
    if center_padding:
        real_output_size = np.round(
            np.asarray(image.GetSize()) * factor + 0.0005
        ).astype(np.uint32)
        diff = ((output_size - real_output_size) * np.asarray(output_spacing)) / 2
        output_origin = np.asarray(image.GetOrigin()) - diff
        # output_origin = output_origin - np.asarray(image.GetSpacing()) / 2 \
        #                 + output_spacing / 2
    else:
        output_origin = np.asarray(image.GetOrigin())

    resampler.SetOutputOrigin(output_origin)
    return resampler.Execute(image)



def normalize_img(img, is_mask=False):
    """
    :param img: numpy image
    :return:
    """
    if not is_mask:
        img[img<-1000] = -1000
        # img = (img - img.min())/(img.max()-img.min())
    else:
        img[img>400]=0
        img[img != 0] = 1
    return img



def preprocess(img_sitk,is_mask=False):
    ori_source, ori_spacing, _ = load_ITK(path_pair[0])
    ori_source = np.flip(sitk.GetArrayFromImage(ori_source), axis=(0))
    ori_target, ori_spacing, _ = load_ITK(path_pair[1])
    ori_target = np.flip(sitk.GetArrayFromImage(ori_target), axis=(0))

    # Pad the one with smaller size
    pad_size = ori_target.shape[0] - ori_source.shape[0]
    if pad_size > 0:
        ori_source = np.pad(ori_source, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=-1024)
    else:
        ori_target = np.pad(ori_target, ((0, -pad_size), (0, 0), (0, 0)), mode='constant', constant_values=-1024)

    assert ori_source.shape == ori_target.shape, "The shape of source and target image should be the same!"

    source, _, _ = resample(ori_source, ori_spacing, spacing)
    source[source < -1024] = -1024
    target, new_spacing, _ = resample(ori_target, ori_spacing, spacing)
    target[target < -1024] = -1024

    if seg_bg:
        bg_hu = np.min(source)
        source_bg_seg, source_bbox = seg_bg_mask(source)
        source[source_bg_seg == 0] = bg_hu

        bg_hu = np.min(target)
        target_bg_seg, source_bbox = seg_bg_mask(target)
        target[target_bg_seg == 0] = bg_hu
        total_voxel = np.prod(target.shape)
        print("##########Area percentage of ROI:{:.2f}, {:.2f}".format(float(np.sum(source_bg_seg)) / total_voxel,
                                                                       float(np.sum(target_bg_seg)) / total_voxel))

    source_seg, _ = seg_lung_mask(source)
    target_seg, _ = seg_lung_mask(target)

    # Pad 0 if shape is smaller than desired size.
    new_origin = np.array((0, 0, 0))
    sz_diff = sz - source.shape
    sz_diff[sz_diff < 0] = 0
    pad_width = [[int(sz_diff[0] / 2), sz_diff[0] - int(sz_diff[0] / 2)],
                 [int(sz_diff[1] / 2), sz_diff[1] - int(sz_diff[1] / 2)],
                 [int(sz_diff[2] / 2), sz_diff[2] - int(sz_diff[2] / 2)]]
    source = np.pad(source, pad_width, constant_values=-1024)
    target = np.pad(target, pad_width, constant_values=-1024)
    source_seg = np.pad(source_seg, pad_width, constant_values=0)
    target_seg = np.pad(target_seg, pad_width, constant_values=0)
    new_origin[sz_diff > 0] = -np.array(pad_width)[sz_diff > 0, 0]

    # Crop if shape is greater than desired size.
    sz_diff = source.shape - sz
    bbox = [[int(sz_diff[0] / 2), int(sz_diff[0] / 2) + sz[0]],
            [int(sz_diff[1] / 2), int(sz_diff[1] / 2) + sz[1]],
            [int(sz_diff[2] / 2), int(sz_diff[2] / 2) + sz[2]]]
    source = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    target = target[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    source_seg = source_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    target_seg = target_seg[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    new_origin[sz_diff > 0] = np.array(bbox)[sz_diff > 0, 0]

    source = normalize_intensity(source)
    target = normalize_intensity(target)
    return source, target, source_seg, target_seg, new_origin, new_spacing
    return sitk_img





def convert_itk_to_support_deepnet(img_sitk, is_mask=False,device=torch.device("cuda:0")):
    img_sz_after_resize = [160]*3
    img_sitk = sitk.GetImageFromArray(sitk.GetArrayFromImage(img_sitk))
    img_after_resize,_ = resize_img(img_sitk,img_sz_after_resize, is_mask=is_mask)
    img_numpy = sitk.GetArrayFromImage(img_after_resize)
    return torch.Tensor(img_numpy.astype(np.float32))[None][None].to(device)


def identity_map(sz,spacing,dtype='float32'):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim==1:
        id = np.mgrid[0:sz[0]]
    elif dim==2:
        id = np.mgrid[0:sz[0],0:sz[1]]
    elif dim==3:
        id = np.mgrid[0:sz[0],0:sz[1],0:sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array( id.astype(dtype) )
    if dim==1:
        id = id.reshape(1,sz[0]) # add a dummy first index

    for d in range(dim):
        id[d]*=spacing[d]

        #id[d]*=2./(sz[d]-1)
        #id[d]-=1.

    # and now store it in a dim+1 array
    if dim==1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0,:] = id[0]
    elif dim==2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0,:, :] = id[0]
        idnp[1,:, :] = id[1]
    elif dim==3:
        idnp = np.zeros([3,sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0,:, :, :] = id[0]
        idnp[1,:, :, :] = id[1]
        idnp[2,:, :, :] = id[2]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    return idnp




def convert_transform_into_itk_bspline(transform,spacing,moving_ref,target_ref):
    if type(transform) == torch.Tensor:
        transform = transform.detach().cpu().numpy()
    cur_trans = transform[0]
    img_sz = np.array(transform.shape[2:])
    moving_spacing_ref = moving_ref.GetSpacing()
    moving_direc_ref = moving_ref.GetDirection()
    moving_orig_ref = moving_ref.GetOrigin()
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
    bias = np.array(target_orig_ref)-np.array(moving_orig_ref)
    bias = -bias.reshape(3,1,1,1)
    transform_physic = cur_trans +bias
    trans = get_transform_with_itk_format(transform_physic,target_spacing_ref, target_orig_ref,target_direc_ref)
    #sitk.WriteTransform(trans, saving_path)
    # Retrive the DField from the Transform
    dfield = trans.GetDisplacementField()
    # Fitting a BSpline from the Deformation Field
    bstx = dfield2bspline(dfield, verbose=True)
    return bstx


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
        ls = sitk.GetArrayFromImage(l_source_itk)
        lt = sitk.GetArrayFromImage(l_target_itk)
        sz_lt = [1, 1] + list(lt.shape)
        l_source = torch.from_numpy(ls[None][None]).to(phi.device)
        if new_phi is None:
            new_phi, new_spacing = resample_image(phi, spacing, sz_lt, 1, zero_boundary=True)
        l_warped = py_utils.compute_warped_image_multiNC(l_source, new_phi, new_spacing, 0, zero_boundary=True)

    id_map = gen_identity_map(warped.shape[2:], resize_factor=1., normalized=True).cuda()
    id_map = (id_map[None] + 1) / 2.
    disp = new_phi - id_map
    bspline_itk = None #convert_transform_into_itk_bspline(disp,new_spacing,source_itk, target_itk)
    return new_phi, warped,l_warped, new_spacing, bspline_itk





def predict(source_itk, target_itk,source_mask_itk=None, target_mask_itk=None, model_path="",device=torch.device("cuda:0")):
    setting_path = "./lung_reg/task_setting.json"
    opt = ParameterDict()
    opt.load_JSON(setting_path)
    source = convert_itk_to_support_deepnet(source_itk,device=device)
    target = convert_itk_to_support_deepnet(target_itk, device=device)
    source_mask = convert_itk_to_support_deepnet(source_mask_itk,is_mask=True) if source_mask_itk is not None else None
    target_mask = convert_itk_to_support_deepnet(target_mask_itk,is_mask=True) if target_mask_itk is not None else None
    network = model(img_sz=[160,160,160],opt=opt)
    network.load_pretrained_model(model_path)
    network.to(device)
    network.train(False)
    with torch.no_grad():
        warped, composed_map, affine_img = network.forward(source, target, source_mask, target_mask)
        inv_warped, composed_inv_map, inv_affine_img = network.forward(target, source, target_mask, source_mask)
    spacing = 1./(np.array(warped.shape[2:])-1)
    del network
    full_composed_map, full_warped,l_full_warped, _, bspline_itk = convert_output_into_itk_support_format(source_itk,target_itk, source_mask_itk, target_mask_itk, composed_map,spacing)
    full_composed_map = full_composed_map.detach().cpu().squeeze().numpy()
    full_inv_composed_map, full_inv_warped,l_full_inv_warped, _, inv_bspline_itk = convert_output_into_itk_support_format(target_itk,source_itk, target_mask_itk, source_mask_itk, composed_inv_map,spacing)
    full_inv_composed_map = full_inv_composed_map.detach().cpu().squeeze().numpy()
    # save_3D_img_from_numpy(full_inv_warped.squeeze().cpu().numpy(),"/playpen-raid1/zyshen/debug/debug_lin_model2.nii.gz",
    #                        source_itk.GetSpacing(), source_itk.GetOrigin(), source_itk.GetDirection())
    # save_3D_img_from_numpy(sitk.GetArrayFromImage(source_itk),
    #                        "/playpen-raid1/zyshen/debug/debug_lin_source.nii.gz",
    #                        source_itk.GetSpacing(), source_itk.GetOrigin(), source_itk.GetDirection())
    return {"phi": full_composed_map, "inv_phi": full_inv_composed_map,"bspline":bspline_itk, "inv_bspline":inv_bspline_itk}



def evaluate_on_dirlab(inv_map,moving_itk, target_itk,dirlab_id):
    MAPPING = {
        "12042G": "copd_000006",
        "12105E": "copd_000007",
        "12109M": "copd_000008",
        "12239Z": "copd_000009",
        "12829U": "copd_000010",
        "13216S": "copd_000001",
        "13528L": "copd_000002",
        "13671Q": "copd_000003",
        "13998W": "copd_000004",
        "17441T": "copd_000005"
    }

    COPD_ID = [
        "copd_000001",
        "copd_000002",
        "copd_000003",
        "copd_000004",
        "copd_000005",
        "copd_000006",
        "copd_000007",
        "copd_000008",
        "copd_000009",
        "copd_000010"
    ]



    def get_dirlab_landmark(case_id):
        assert case_id in COPD_ID
        exp_landmark_path = os.path.join(dirlab_landmarks_folder, case_id + "_EXP.vtk")
        insp_landmark_path = os.path.join(dirlab_landmarks_folder, case_id + "_INSP.vtk")
        exp_landmark = read_vtk(exp_landmark_path)["points"]
        insp_landmark = read_vtk(insp_landmark_path)["points"]
        return exp_landmark, insp_landmark

    def read_vtk(path):
        data = pv.read(path)
        data_dict = {}
        data_dict["points"] = data.points.astype(np.float32)
        data_dict["faces"] = data.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        for name in data.array_names:
            try:
                data_dict[name] = data[name]
            except:
                pass
        return data_dict

    def warp_points(points, inv_map, moving_itk, target_itk):
        """
        in easyreg the inv transform coord is from [0,1], so here we need to read mesh in voxel coord and then normalized it to [0,1],
        the last step is to transform warped mesh into word/ voxel coord
        the transformation map use default [0,1] coord unless the ref img is provided
        here the transform map is  in inversed voxel space or in  inversed physical space ( width,height, depth)
        but the points should be in standard voxel/itk space (depth, height, width)
        :return:
        """

        import numpy as np
        import torch.nn.functional as F
        # first make everything in voxel coordinate, depth, height, width
        img_sz = np.array(inv_map.shape[1:])
        standard_spacing = 1 / (img_sz - 1)  # width,height, depth
        standard_spacing = np.flipud(standard_spacing)  # depth, height, width
        moving_img = moving_itk
        moving_spacing = moving_img.GetSpacing()
        moving_spacing = np.array(moving_spacing)
        moving_origin = moving_img.GetOrigin()
        moving_origin = np.array(moving_origin)

        target_img = target_itk
        target_spacing = target_img.GetSpacing()
        target_spacing = np.array(target_spacing)
        target_origin = target_img.GetOrigin()
        target_origin = np.array(target_origin)

        moving = sitk.GetArrayFromImage(moving_img)
        slandmark_index = (points-moving_origin) / moving_spacing
        for coord in slandmark_index:
            coord_int  = [int(c) for c in coord]
            moving[coord_int[2],coord_int[1],coord_int[0]] = 2.
        save_3D_img_from_numpy(moving,"/playpen-raid2/zyshen/debug/{}_padded.nii.gz".format(dirlab_id+"_moving"),
                               spacing=moving_img.GetSpacing(), orgin=moving_img.GetOrigin(), direction=moving_img.GetDirection())

        points = (points - moving_origin) / moving_spacing * standard_spacing
        points = points * 2 - 1
        grid_sz = [1] + [points.shape[0]] + [1, 1, 3]  # 1*N*1*1*3
        grid = points.reshape(*grid_sz)
        grid = torch.Tensor(grid).cuda()
        inv_map = torch.Tensor(inv_map).cuda()
        inv_map_sz = [1, 3] + list(img_sz)  # width,height, depth
        inv_map = inv_map.view(*inv_map_sz)  # 1*3*X*Y*Z
        points_wraped = F.grid_sample(inv_map, grid, mode='bilinear', padding_mode='border',
                                      align_corners=True)  # 1*3*N*1*1
        points_wraped = points_wraped.detach().cpu().numpy()
        points_wraped = np.transpose(np.squeeze(points_wraped))
        points_wraped = np.flip(points_wraped, 1) / standard_spacing * target_spacing + target_origin

        warp = sitk.GetArrayFromImage(target_img)
        wlandmark_index = (points_wraped - target_origin) / target_spacing
        for coord in wlandmark_index:
            coord_int = [int(c) for c in coord]
            warp[coord_int[2], coord_int[1], coord_int[0]] = 2.
        save_3D_img_from_numpy(warp, "/playpen-raid2/zyshen/debug/{}_debug.nii.gz".format("warp"))

        return points_wraped


    assert dirlab_id in MAPPING, "{} doesn't belong to ten dirlab cases:{}".format(dirlab_id, MAPPING.keys())
    exp_landmark, insp_landmark = get_dirlab_landmark(MAPPING[dirlab_id])
    warped_landmark = warp_points(exp_landmark, inv_map, moving_itk, target_itk)
    diff = warped_landmark - insp_landmark
    diff_norm = np.linalg.norm(diff, ord=2, axis=1)
    print("before register landmark error norm: {}".format(
        np.linalg.norm(exp_landmark - insp_landmark, ord=2, axis=1).mean()))
    print("after register landmark error norm: {}".format(diff_norm.mean()))
    

def get_file_name(img_path):
    get_fn = lambda x: os.path.split(x)[-1]
    file_name = get_fn(img_path).split(".")[0]
    return file_name

if __name__ == "__main__":
    source_path = "/playpen-raid1/Data/DIRLABCasesHighRes/12042G_EXP_STD_USD_COPD.nrrd"
    target_path = "/playpen-raid1/Data/DIRLABCasesHighRes/12042G_INSP_STD_USD_COPD.nrrd"
    source_mask_path = "/playpen-raid1/lin.tian/data/raw/DIRLABCasesHighRes/copd6/copd6_EXP_label.nrrd"
    target_mask_path = "/playpen-raid1/lin.tian/data/raw/DIRLABCasesHighRes/copd6/copd6_INSP_label.nrrd"
    model_path = "./lung_reg/lin_model"
    source_itk = sitk.ReadImage(source_path)
    target_itk = sitk.ReadImage(target_path)
    source_mask_itk = sitk.ReadImage(source_mask_path) if len(source_mask_path) else None
    target_mask_itk = sitk.ReadImage(target_mask_path) if len(target_mask_path) else None
    preprocessed_source_itk = preprocess(source_itk)
    preprocessed_target_itk = preprocess(target_itk)
    #sitk.WriteImage(preprocessed_source_itk,"/playpen-raid1/zyshen/debug/12042G_preprocessed.nii.gz")
    if source_mask_itk is not None and target_mask_itk is not None:
        preprocessed_source_mask_itk = preprocess(source_mask_itk,is_mask=True)
        preprocessed_target_mask_itk = preprocess(target_mask_itk,is_mask=True)
    else:
        preprocessed_source_mask_itk = None
        preprocessed_target_mask_itk = None
    output_dict = predict(preprocessed_source_itk, preprocessed_target_itk,preprocessed_source_mask_itk,preprocessed_target_mask_itk,
                          model_path=model_path)

    dirlab_id = get_file_name(source_path).split("_")[0]
    evaluate_on_dirlab(output_dict["inv_phi"], preprocessed_source_itk, preprocessed_target_itk, dirlab_id)