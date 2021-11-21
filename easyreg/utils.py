import torch
import numpy as np
import skimage
import os
import torchvision.utils as utils
import SimpleITK as sitk
from skimage import color
import mermaid.image_sampling as py_is
from mermaid.data_wrapper import AdaptVal,MyTensor
from .net_utils import gen_identity_map
from .net_utils import Bilinear
import mermaid.utils as py_utils
import mermaid.module_parameters as pars
import mermaid.smoother_factory as sf

def get_reg_pair(data,ch=1):
    """
    get image pair from data, pair is concatenated by the channel
    :param data: a dict, including {'img':, 'label':}
    :param pair:
    :param target: target image
    :param ch: the num of input channel
    :return: image BxCxXxYxZ, label BxCxXxYxZ
    """
    if 'label' in data:
        return data['image'][:,0:ch], data['image'][:,ch:2*ch],data['label'][:,0:ch],data['label'][:,ch:2*ch]
    else:
        return data['image'][:,0:ch], data['image'][:,ch:2*ch],None, None


def get_seg_pair(data, is_train=True):
    """
    get image and gt from data, pair is concatenated by the channel
    :param data: a dict, including {'img':, 'label':}
    :return: image BxCxXxYxZ, label BxCxXxYxZ
    """
    if not is_train:
        data['image']= data['image'][0]
        if 'label' in data:
            data['label'] = data['label'][0]

    if 'label' in data:
        return data['image'], data['label']
    else:
        return data['image'],None



def sigmoid_explode(ep, static =5, k=5):
    """
    factor  increase with epoch, factor = (k + exp(ep / k))/k
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the explode factor
    :return:
    """
    static = static
    if ep < static:
        return 1.
    else:
        ep = ep - static
        factor= (k + np.exp(ep / k))/k
        return float(factor)

def sigmoid_decay(ep, static =5, k=5):
    """
    factor  decease with epoch, factor = k/(k + exp(ep / k))
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the decay factor
    :return:
    """
    static = static
    if ep < static:
        return float(1.)
    else:
        ep = ep - static
        factor =  k/(k + np.exp(ep / k))
        return float(factor)


def factor_tuple(input,factor):
    """
    multiply a factor to each tuple elem
    :param input:
    :param factor:
    :return:
    """
    input_np = np.array(list(input))
    input_np = input_np*factor
    return tuple(list(input_np))
def resize_spacing(img_sz,img_sp,factor):
    """
    compute the new spacing with regard to the image resampling factor
    :param img_sz: img sz
    :param img_sp: img spacing
    :param factor: factor of resampling image
    :return:
    """
    img_sz_np = np.array(list(img_sz))
    img_sp_np = np.array(list(img_sp))
    new_sz_np = img_sz_np*factor
    new_sp = img_sp_np*(img_sz_np-1)/(new_sz_np-1)
    return tuple(list(new_sp))


def save_image_with_scale(path, variable):
    """
    the input variable is [-1,1], save into image
    :param path: path to save
    :param variable: variable to save, XxY
    :return:
    """

    arr = variable.cpu().data.numpy()
    arr = np.clip(arr, -1., 1.)
    arr = (arr+1.)/2 * 255.
    arr = arr.astype(np.uint8)
    skimage.io.imsave(path, arr)


def get_transform_with_itk_format(disp_np, spacing,original, direction):
    import SimpleITK as sitk
    # Create a composite transform then write and read.
    displacement = sitk.DisplacementFieldTransform(3)
    field_size = list(np.flipud(disp_np.shape[1:]).astype(np.float64))
    field_origin = list(original)
    field_spacing = list(spacing)
    field_direction = list(direction)  # direction cosine matrix (row major order)

    # Concatenate all the information into a single list.
    displacement.SetFixedParameters(field_size + field_origin + field_spacing + field_direction)
    displacement.SetParameters(np.transpose(disp_np,[1,2,3,0]).reshape(-1).astype(np.float64))
    return displacement




def make_image_summary(images, truths, raw_output, maxoutput=4, overlap=True):
    """make image summary for tensorboard

    :param images: torch.Variable, NxCxDxHxW, 3D image volume (C:channels)
    :param truths: torch.Variable, NxDxHxW, 3D label mask
    :param raw_output: torch.Variable, NxCxHxWxD: prediction for each class (C:classes)
    :param maxoutput: int, number of samples from a batch
    :param overlap: bool, overlap the image with groundtruth and predictions
    :return: summary_images: list, a maxoutput-long list with element of tensors of Nx
    """
    slice_ind = images.size()[2] // 2
    images_2D = images.data[:maxoutput, :, slice_ind, :, :]
    truths_2D = truths.data[:maxoutput, slice_ind, :, :]
    predictions_2D = torch.max(raw_output.data, 1)[1][:maxoutput, slice_ind, :, :]

    grid_images = utils.make_grid(images_2D, pad_value=1)
    grid_truths = utils.make_grid(labels2colors(truths_2D, images=images_2D, overlap=overlap), pad_value=1)
    grid_preds = utils.make_grid(labels2colors(predictions_2D, images=images_2D, overlap=overlap), pad_value=1)

    return torch.cat([grid_images, grid_truths, grid_preds], 1)


def labels2colors(labels, images=None, overlap=False):
    """Turn label masks into color images
    :param labels: torch.tensor, NxMxN
    :param images: torch.tensor, NxMxN or NxMxNx3
    :param overlap: bool
    :return: colors: torch.tensor, Nx3xMxN
    """
    colors = []
    if overlap:
        if images is None:
            raise ValueError("Need background images when overlap is True")
        else:
            for i in range(images.size()[0]):
                image = images.squeeze()[i, :, :]
                label = labels[i, :, :]
                colors.append(color.label2rgb(label.cpu().numpy(), image.cpu().numpy(), bg_label=0, alpha=0.7))
    else:
        for i in range(images.size()[0]):
            label = labels[i, :, :]
            colors.append(color.label2rgb(label.numpy(), bg_label=0))

    return torch.Tensor(np.transpose(np.stack(colors, 0), (0, 3, 1, 2))).cuda()





def t2np(v):
    """
    Takes a torch array and returns it as a numpy array on the cpu

    :param v: torch array
    :return: numpy array
    """

    if type(v) == torch.Tensor:
        return v.detach().cpu().numpy()
    else:
        try:
            return v.cpu().numpy()
        except:
            return v



def make_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist



def lift_to_dimension(A,dim):
    """
    Creates a view of A of dimension dim (by adding dummy dimensions if necessary).
    Assumes a numpy array as input

    :param A: numpy array
    :param dim: desired dimension of view
    :return: returns view of A of appropriate dimension
    """

    current_dim = len(A.shape)
    if current_dim>dim:
        raise ValueError('Can only add dimensions, but not remove them')

    if current_dim==dim:
        return A
    else:
        return A.reshape([1]*(dim-current_dim)+list(A.shape))





def update_affine_param( cur_af, last_af): # A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2
    """
       update the current affine parameter A2 based on last affine parameter A1
        A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2, results in the composed affine parameter A3=(A2A1, A2*b1+b2)
       :param cur_af: current affine parameter
       :param last_af: last affine parameter
       :return: composed affine parameter A3
    """
    cur_af = cur_af.view(cur_af.shape[0], 4, 3)
    last_af = last_af.view(last_af.shape[0],4,3)
    updated_af = torch.zeros_like(cur_af.data).to(cur_af.device)
    dim =3
    updated_af[:,:3,:] = torch.matmul(cur_af[:,:3,:],last_af[:,:3,:])
    updated_af[:,3,:] = cur_af[:,3,:] + torch.squeeze(torch.matmul(cur_af[:,:3,:], torch.transpose(last_af[:,3:,:],1,2)),2)
    updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
    return updated_af

def get_inverse_affine_param(affine_param,dim=3):
    """A2(A1*x+b1) +b2= A2A1*x + A2*b1+b2 = x    A2= A1^-1, b2 = - A2^b1"""

    affine_param = affine_param.view(affine_param.shape[0], dim+1, dim)
    inverse_param = torch.zeros_like(affine_param.data).to(affine_param.device)
    for n in range(affine_param.shape[0]):
        tm_inv = torch.inverse(affine_param[n, :dim,:])
        inverse_param[n, :dim, :] = tm_inv
        inverse_param[n, dim, :] = - torch.matmul(tm_inv, affine_param[n, dim, :])
    inverse_param = inverse_param.contiguous().view(affine_param.shape[0], -1)
    return inverse_param

def gen_affine_map(Ab, img_sz, dim=3):
    """
       generate the affine transformation map with regard to affine parameter
       :param Ab: affine parameter
       :param img_sz: image sz  [X,Y,Z]
       :return: affine transformation map
    """
    Ab = Ab.view(Ab.shape[0], dim+1, dim)
    phi = gen_identity_map(img_sz).to(Ab.device)
    phi_cp = phi.view(dim, -1)
    affine_map = torch.matmul(Ab[:, :dim, :], phi_cp)
    affine_map = Ab[:, dim, :].contiguous().view(-1, dim, 1) + affine_map
    affine_map = affine_map.view([Ab.shape[0]] + list(phi.shape))
    return affine_map

def transfer_mermaid_affine_into_easyreg_affine(affine_param, dim=3):
    affine_param = affine_param.view(affine_param.shape[0], dim+1, dim)
    I = torch.ones(dim).to(affine_param.device)
    b = affine_param[:, dim,:]
    affine_param[:,:dim,:]=  affine_param[:,:dim,:].transpose(1, 2)
    affine_param[:, dim,:] =2*b +torch.matmul(affine_param[:,:dim,:],I)-1 # easyreg assume map is defined in [-1,1] whle the mermaid assumes [0,1]
    affine_param = affine_param.contiguous()
    affine_param = affine_param.view(affine_param.shape[0],-1)
    return affine_param

def transfer_easyreg_affine_into_mermaid_affine(affine_param, dim=3):
    affine_param = affine_param.view(affine_param.shape[0], dim+1, dim)
    I = torch.ones(dim).to(affine_param.device)
    b = affine_param[:, dim,:]
    affine_param[:, dim,:] = (b-torch.matmul(affine_param[:,:dim,:],I)+1)/2 # the order here is important
    affine_param[:,:dim,:]=  affine_param[:,:dim,:].transpose(1, 2)
    affine_param = affine_param.contiguous()
    affine_param = affine_param.view(affine_param.shape[0],-1)
    return affine_param

def save_affine_param_with_easyreg_custom(affine_param, output_path, fname_list, affine_compute_from_mermaid=False):
    if affine_param is not None:
        affine_param = affine_param.detach().clone()
        if affine_compute_from_mermaid:
            affine_param = transfer_mermaid_affine_into_easyreg_affine(affine_param)

        if isinstance(affine_param, list):
            affine_param = affine_param[0]
        affine_param = affine_param.detach().cpu().numpy()
        for i in range(len(fname_list)):
            np.save(os.path.join(output_path, fname_list[i]) + '_affine_param.npy', affine_param[i])


def get_warped_img_map_param( Ab, img_sz, moving, dim=3, zero_boundary=True):
    """
           generate the affine transformation map with regard to affine parameter
           :param Ab: affine parameter
           :param img_sz: image sz [X,Y,Z]
           :param moving:  moving image BxCxXxYxZ
           :param zero_boundary:  zero_boundary condition
           :return: affine image, affine transformation map, affine parameter
        """
    bilinear = Bilinear(zero_boundary)
    affine_map = gen_affine_map(Ab,img_sz,dim)
    output = bilinear(moving, affine_map)
    return output, affine_map, Ab



def show_current_pair_by_3d_slice(iS,iT):
    """
    visualize the pair image by slice
    :param iS: source image
    :param iT: target image
    :return:
    """
    import matplotlib.pyplot as plt
    import easyreg.viewers as viewers
    fig, ax = plt.subplots(2,3)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    ivsx = viewers.ImageViewer3D_Sliced(ax[0][0], iS, 0, 'source X', True)
    ivsy = viewers.ImageViewer3D_Sliced(ax[0][1], iS, 1, 'source Y', True)
    ivsz = viewers.ImageViewer3D_Sliced(ax[0][2], iS, 2, 'source Z', True)

    ivtx = viewers.ImageViewer3D_Sliced(ax[1][0], iT, 0, 'target X', True)
    ivty = viewers.ImageViewer3D_Sliced(ax[1][1], iT, 1, 'target Y', True)
    ivtz = viewers.ImageViewer3D_Sliced(ax[1][2], iT, 2, 'target Z', True)


    feh = viewers.FigureEventHandler(fig)
    feh.add_axes_event('button_press_event', ax[0][0], ivsx.on_mouse_press, ivsx.get_synchronize, ivsx.set_synchronize)
    feh.add_axes_event('button_press_event', ax[0][1], ivsy.on_mouse_press, ivsy.get_synchronize, ivsy.set_synchronize)
    feh.add_axes_event('button_press_event', ax[0][2], ivsz.on_mouse_press, ivsz.get_synchronize, ivsz.set_synchronize)

    feh.add_axes_event('button_press_event', ax[1][0], ivtx.on_mouse_press, ivtx.get_synchronize, ivtx.set_synchronize)
    feh.add_axes_event('button_press_event', ax[1][1], ivty.on_mouse_press, ivty.get_synchronize, ivty.set_synchronize)
    feh.add_axes_event('button_press_event', ax[1][2], ivtz.on_mouse_press, ivtz.get_synchronize, ivtz.set_synchronize)

    feh.synchronize([ax[0][0], ax[1][0]])
    feh.synchronize([ax[0][1], ax[1][1]])
    feh.synchronize([ax[0][2], ax[1][2]])



def get_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) :
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        if not isinstance(factor, list):
            lowResSize[2::] = (np.ceil((np.array(sz[2:]) * factor))).astype('int16')
        else:
            lowResSize[2::] = (np.ceil((np.array(sz[2:]) * np.array(factor)))).astype('int16')

        if lowResSize[-1]%2!=0:
            lowResSize[-1]-=1
            print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize


def get_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    if len(sz) == len(spacing):
        sz = [1,1]+sz
    if len(lowResSize)==len(spacing):
        lowResSize = [1,1]+lowResSize
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)

def _compute_low_res_image(I,spacing,low_res_size,zero_boundary=False):
    sampler = py_is.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],1,zero_boundary=zero_boundary)
    return low_res_image


def resample_image(I,spacing,desiredSize, spline_order=1,zero_boundary=False,identity_map=None):
    """
    Resample an image to a given desired size

    :param I: Input image (expected to be of BxCxXxYxZ format)
    :param spacing: array describing the spatial spacing
    :param desiredSize: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
    :return: returns a tuple: the downsampled image, the new spacing after downsampling
    """
    if len(I.shape) != len(desiredSize)+2:
        desiredSize = desiredSize[2:]
    sz = np.array(list(I.size()))
    # check that the batch size and the number of channels is the same
    nrOfI = sz[0]
    nrOfC = sz[1]

    desiredSizeNC = np.array([nrOfI,nrOfC]+list(desiredSize))

    newspacing = spacing*((sz[2::].astype('float')-1.)/(desiredSizeNC[2::].astype('float')-1.)) ###########################################
    if identity_map is not None:
        idDes= identity_map
    else:
        idDes = torch.from_numpy(py_utils.identity_map_multiN(desiredSizeNC,newspacing)).to(I.device)
    # now use this map for resampling
    ID = py_utils.compute_warped_image_multiNC(I, idDes, newspacing, spline_order,zero_boundary)

    return ID, newspacing


def get_resampled_image(I,spacing,desiredSize, spline_order=1,zero_boundary=False,identity_map=None):
    """

    :param I:  B C X Y Z
    :param spacing: spx spy spz
    :param desiredSize: B C X Y Z
    :param spline_order:
    :param zero_boundary:
    :param identity_map:
    :return:
    """
    if spacing is None:
        img_sz = I.shape[2:]
        spacing = 1./(np.array(img_sz)-1)
    if identity_map is not None:# todo  will remove, currently fix for symmetric training
        if I.shape[0] != identity_map.shape[0]:
            n_batch = I.shape[0]
            desiredSize =desiredSize.copy()
            desiredSize[0] = n_batch
            identity_map =  identity_map[:n_batch]
    resampled,new_spacing = resample_image(I, spacing, desiredSize, spline_order=spline_order, zero_boundary=zero_boundary,identity_map=identity_map)
    return resampled



def load_inital_weight_from_pt(path):
    init_weight = torch.load(path)
    return init_weight



def get_init_weight_from_label_map(lsource, spacing,default_multi_gaussian_weights,multi_gaussian_weights,weight_type='w_K_w'):
    """
    for rdmm model with spatial-variant regularizer, we initialize multi gaussian weight with regard to the label map
    assume img sz BxCxXxYxZ and N gaussian smoothers are taken, the return weight map should be BxNxXxYxZ
    :param lsource: label of the source image
    :param spacing: image spacing
    :param default_multi_gaussian_weights: multi-gaussian weight set for the background
    :param multi_gaussian_weights: multi-gaussian weight set for the foreground( labeled region)
    :param weight_type: either w_K_w or sqrt_w_K_sqrt_w
    :return: weight map BxNxXxYxZ
    """
    if type(lsource)==torch.Tensor:
        lsource = lsource.detach().cpu().numpy()
    sz = lsource.shape[2:]
    nr_of_mg_weights = len(default_multi_gaussian_weights)
    sh_weights = [lsource.shape[0]] + [nr_of_mg_weights] + list(sz)
    weights = np.zeros(sh_weights, dtype='float32')
    for g in range(nr_of_mg_weights):
        weights[:, g, ...] = default_multi_gaussian_weights[g]
    indexes = np.where(lsource>0)
    for g in range(nr_of_mg_weights):
        weights[indexes[0], g, indexes[2], indexes[3],indexes[4]] = np.sqrt(multi_gaussian_weights[g]) if weight_type=='w_K_w' else multi_gaussian_weights[g]
    weights = MyTensor(weights)
    local_smoother  = get_single_gaussian_smoother(0.02,sz,spacing)
    sm_weight = local_smoother.smooth(weights)
    return sm_weight


def get_single_gaussian_smoother(gaussian_std,sz,spacing):
    s_m_params = pars.ParameterDict()
    s_m_params['smoother']['type'] = 'gaussian'
    s_m_params['smoother']['gaussian_std'] = gaussian_std
    s_m = sf.SmootherFactory(sz, spacing).create_smoother(s_m_params)
    return s_m



def get_gaussion_weight_from_tsk_opt(opt):
    return opt['']


def normalize_spacing(spacing,sz,silent_mode=False):
    """
    Normalizes spacing.
    :param spacing: Vector with spacing info, in XxYxZ format
    :param sz: size vector in XxYxZ format
    :return: vector with normalized spacings in XxYxZ format
    """
    dim = len(spacing)
    # first determine the largest extent
    current_largest_extent = -1
    extent = np.zeros_like(spacing)
    for d in range(dim):
        current_extent = spacing[d]*(sz[d]-1)
        extent[d] = current_extent
        if current_extent>current_largest_extent:
            current_largest_extent = current_extent

    scalingFactor = 1./current_largest_extent
    normalized_spacing = spacing*scalingFactor

    normalized_extent = extent*scalingFactor

    if not silent_mode:
        print('Normalize spacing: ' + str(spacing) + ' -> ' + str(normalized_spacing))
        print('Normalize spacing, extent: ' + str(extent) + ' -> ' + str(normalized_extent))

    return normalized_spacing




def dfield2bspline(dfield, sp_order=3, n_nodes=50, verbose=False):
    # BSpline configuration
    dim = dfield.GetDimension()
    n_nodes = np.full(dim, n_nodes)
    mesh_sz = n_nodes - sp_order
    physical_dim = np.array(dfield.GetSpacing()) * (np.array(dfield.GetSize()) - 1)

    # This transform is used to compute the origin and spacing properly
    bstx = sitk.BSplineTransform(sp_order)
    bstx.SetTransformDomainOrigin(dfield.GetOrigin())
    bstx.SetTransformDomainPhysicalDimensions(physical_dim.tolist())
    bstx.SetTransformDomainMeshSize(mesh_sz.tolist())
    bstx.SetTransformDomainDirection(dfield.GetDirection())

    if verbose:
        print('Adjusting BSpline to the Displacement field...')
    Idf = sitk.GetArrayViewFromImage(dfield)
    img_params = []

    for i in range(dim):
        # Create the image for dim i
        dfi = sitk.GetImageFromArray(Idf[..., i])
        dfi.SetDirection(dfield.GetDirection())
        dfi.SetOrigin(dfield.GetOrigin())
        dfi.SetSpacing(dfield.GetSpacing())
        # Downsampling the image field to the desired BSpline
        downsampler = sitk.ResampleImageFilter()
        downsampler.SetInterpolator(sitk.sitkBSpline)
        downsampler.SetDefaultPixelValue(0)
        # By default the Identity is used as transform
        downsampler.SetSize(bstx.GetCoefficientImages()[i].GetSize())
        downsampler.SetOutputSpacing(bstx.GetCoefficientImages()[i].GetSpacing())
        downsampler.SetOutputOrigin(bstx.GetCoefficientImages()[i].GetOrigin())
        downsampler.SetOutputDirection(dfield.GetDirection())
        out = downsampler.Execute(dfi)

        decomp = sitk.BSplineDecompositionImageFilter()
        decomp.SetSplineOrder(sp_order)

        img_params.append(decomp.Execute(out))

    bstx = sitk.BSplineTransform(img_params, sp_order)
    return bstx


    # dtransform = sitk.ReadTransform(df_name)
    # # Retrive the DField from the Transform
    # dfield = sitk.DisplacementFieldTransform(dtransform).GetDisplacementField()
    # # Fitting a BSpline from the Deformation Field
    # bstx = dfield2bspline(dfield, verbose=True)
    #
    # # Save the BSpline Transform
    # sitk.WriteTransform(bstx, df_name.replace('_disp.h5', '_disp_bs.tfm'))