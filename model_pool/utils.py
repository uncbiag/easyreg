import torch
import numpy as np
import skimage
import torch.nn as nn
import os
import torchvision.utils as utils
from skimage import color

from model_pool.net_utils import gen_identity_map
from functions.bilinear import Bilinear


def get_pair(data, pair= True, target=None):
    if 'label' in data:
        return data['image'][:,0:1], data['image'][:,1:2],data['label'][:,0:1],data['label'][:,1:2]
    else:
        return data['image'][:,0:1], data['image'][:,1:2],None, None


def sigmoid_explode(ep, static =5, k=5):
    static = static
    if ep < static:
        return 1.
    else:
        ep = ep - static
        factor= (k + np.exp(ep / k))/k
        return float(factor)

def sigmoid_decay(ep, static =5, k=5):
    static = static
    if ep < static:
        return float(1.)
    else:
        ep = ep - static
        factor =  k/(k + np.exp(ep / k))
        return float(factor)


def factor_tuple(input,factor):
    input_np = np.array(list(input))
    input_np = input_np*factor
    return tuple(list(input_np))

def organize_data(moving, target, sched='depth_concat'):
    if sched == 'depth_concat':
        input = torch.cat([moving, target], dim=1)
    elif sched == 'width_concat':
        input = torch.cat((moving, target), dim=3)
    elif sched =='list_concat':
        input = torch.cat((moving.unsqueeze(0),target.unsqueeze(0)),dim=0)
    return input


def save_image_with_scale(path, variable):
      arr = variable.cpu().data.numpy()
      arr = np.clip(arr, -1., 1.)
      arr = (arr+1.)/2 * 255.
      arr = arr.astype(np.uint8)
      skimage.io.imsave(path, arr)


def save_result(path, appendix, moving, target, reproduce):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(moving.size(0)):
      save_image_with_scale(path+appendix+"_b{:02d}_moving.tif".format(i), moving[i,0,...])
      save_image_with_scale(path+appendix+"_b{:02d}_target.tif".format(i), target[i,0,...])
      save_image_with_scale(path+appendix+"_b{:02d}_reproduce.tif".format(i),reproduce[i,0,...])


def unet_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if not m.weight is None:
            nn.init.xavier_normal(m.weight.data)
        if not m.bias is None:
            nn.init.xavier_normal(m.bias.data)

def vnet_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.mkdir(path)
    prefix_save = os.path.join(path, prefix)
    name = '_'.join([prefix_save, filename])
    torch.save(state, name)
    if is_best:
        torch.save(state, path + '/model_best.pth.tar')

def CrossCorrelationLoss(input, target):
    eps =1e-9
    size_img = input.size()
    input = input.view(size_img[0], size_img[1],-1)
    target = target.view(size_img[0], size_img[1], -1)
    m_input = input - torch.mean(input,dim=2, keepdim = True)
    m_target = target - torch.mean(target, dim=2, keepdim = True)
    cc = torch.sum(m_input * m_target, dim=2, keepdim=True)
    norm = torch.sqrt(torch.sum(m_input**2, dim=2, keepdim=True)) * torch.sqrt(torch.sum(m_target**2, dim=2, keepdim=True))
    ncc = cc/(norm +eps)
    ncc = - torch.sum(ncc)/(size_img[0] * size_img[1])
    return ncc





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
    cur_af = cur_af.view(cur_af.shape[0], 4, 3)
    last_af = last_af.view(last_af.shape[0],4,3)
    updated_af = torch.zeros_like(cur_af.data).cuda()
    dim =3
    updated_af[:,:3,:] = torch.matmul(cur_af[:,:3,:],last_af[:,:3,:])
    updated_af[:,3,:] = cur_af[:,3,:] + torch.squeeze(torch.matmul(cur_af[:,:3,:], torch.transpose(last_af[:,3:,:],1,2)),2)
    updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
    return updated_af

def get_inverse_affine_param(affine_param):
    """A2(A1*x+b1) +b2= A2A1*x + A2*b1+b2 = x    A2= A1^-1, b2 = - A2^b1"""

    affine_param = affine_param.view(affine_param.shape[0], 4, 3)
    inverse_param = torch.zeros_like(affine_param.data).cuda()
    for n in range(affine_param.shape[0]):
        tm_inv = torch.inverse(affine_param[n, :3,:])
        inverse_param[n, :3, :] = tm_inv
        inverse_param[n, 3, :] = - torch.matmul(tm_inv, affine_param[n, 3, :])
    inverse_param = inverse_param.contiguous().view(affine_param.shape[0], -1)
    return inverse_param

def gen_affine_map(Ab, img_sz, dim=3):
    Ab = Ab.view(Ab.shape[0], 4, 3)  # 3d: (batch,3)
    phi = gen_identity_map(img_sz)
    phi_cp = phi.view(dim, -1)
    affine_map = None
    if dim == 3:
        affine_map = torch.matmul(Ab[:, :3, :], phi_cp)
        affine_map = Ab[:, 3, :].contiguous().view(-1, 3, 1) + affine_map
        affine_map = affine_map.view([Ab.shape[0]] + list(phi.shape))
    return affine_map

def get_warped_img_map_param( Ab, img_sz, moving, dim=3, zero_boundary=True):
    bilinear = Bilinear(zero_boundary)
    affine_map = gen_affine_map(Ab,img_sz,dim)
    output = bilinear(moving, affine_map)
    return output, affine_map, Ab



def show_current_images_3d(iS,iT):
    import matplotlib.pyplot as plt
    import model_pool.viewers as viewers
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

