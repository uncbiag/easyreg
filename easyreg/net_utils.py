import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import torch.nn.init as init
import os
from easyreg.reproduce_paper_results import reproduce_paper_result

dim = 3
Conv = nn.Conv2d if dim == 2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
conv = F.conv2d if dim == 2 else F.conv3d


class conv_bn_rel(nn.Module):
    """
    conv + bn (optional) + relu

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, group=1, dilation=1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1, dilation=1)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1,
                                      dilation=1)

        self.bn = BatchNorm(out_channels) if bn else None #, eps=0.0001, momentum=0, affine=True
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class FcRel(nn.Module):
    """
    fc+ relu(option)
    """
    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x



class Bilinear(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False, using_scale=True):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.using_scale = using_scale
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]),
                                                     padding_mode=self.zero_boundary, align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if input1.shape[0] != input2.shape[0]:
            n_batch = input1.shape[0]
            input2 = input2[:n_batch]

        if self.using_scale:

            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output







def identity_map_for_reproduce(sz):
    """
    Returns an identity map. todo keep for preproduce result, this function will be disabled in the next release, replaced by spacing version

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[-1:1.:2. / sz[0]]
    elif dim == 2:
        id = np.mgrid[-1.:1.:2. / sz[0], -1.:1.:2. / sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[-1.:1.:2. / sz[0], -1.:1.:2. / sz[1], -1.:1.:2. / sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))

def identity_map(sz, dtype= np.float32):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1./ (np.array(sz)-1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d]*2 - 1

    return torch.from_numpy(id.astype(np.float32))


def not_normalized_identity_map(sz):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0: sz[0]]
    elif dim == 2:
        id = np.mgrid[0: sz[0], 0: sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0: sz[0], 0:sz[1], 0: sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))


def gen_identity_map(img_sz, resize_factor=1.,normalized=True):
    """
    given displacement field,  add displacement on grid field  todo  now keep for reproduce  this function will be disabled in the next release, replaced by spacing version
    """
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(dim)]
    if normalized:
        grid = identity_map(img_sz) if not reproduce_paper_result else identity_map_for_reproduce(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid


class AffineConstrain(object):
    """
    regularization on affine parameters,
    sched:
        'l2':  square loss
        'det': determinant loss
    """
    def __init__(self):
        if dim == 3:
            self.affine_identity = torch.zeros(12).cuda()
            self.affine_identity[0] = 1.
            self.affine_identity[4] = 1.
            self.affine_identity[8] = 1.
        else:
            raise ValueError("Not Implemented")

    def __call__(self, affine_param, sched='l2'):
        if sched == 'l2':
            return (self.affine_identity - affine_param) ** 2
        elif sched == 'det':
            mean_det = 0.
            for i in range(affine_param.shape[0]):
                affine_matrix = affine_param[i, :9].contiguous().view(3, 3)
                mean_det += torch.det(affine_matrix)
            return mean_det / affine_param.shape[0]




def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def space_normal(tensors, std=0.1):
    """
    space normalize for the net kernel
    :param tensor:
    :param mean:
    :param std:
    :return:
    """
    if isinstance(tensors, torch.Tensor):
        space_normal(tensors.data, std=std)
        return tensors
    for n in range(tensors.size()[0]):
        for c in range(tensors.size()[1]):
            dim = tensors[n][c].dim()
            sz = tensors[n][c].size()
            mus = np.zeros(dim)
            stds = std * np.ones(dim)
            print('WARNING: What should the spacing be here? Needed for new identity map code')
            raise ValueError('Double check the spacing here before running this code')
            spacing = np.ones(dim)
            centered_id = centered_identity_map(sz, spacing)
            g = compute_normalized_gaussian(centered_id, mus, stds)
            tensors[n, c] = torch.from_numpy(g)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.038, 0.042)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        space_normal(m.weight.data)
    elif classname.find('Linear') != -1:
        space_normal(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_rd_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'rd_normal':
        net.apply(weights_init_rd_normal)
    elif init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'uniform':
        net.apply(weights_init_uniform)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def print_network(net):
    """ print out the structure of the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def resume_train(model_path, model, optimizer):
    """
    resume the training from checkpoint
    :param model_path: the checkpoint path
    :param model: the model to be set
    :param optimizer: the optimizer to be set
    :return:
    """
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')  # {'cuda:'+str(old_gpu):'cuda:'+str(cur_gpu)})
        start_epoch = 0
        best_prec1 = 0.0
        load_only_one = False
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print("the started epoch now is {}".format(start_epoch))
        else:
            start_epoch = 0
        if 'best_loss' in checkpoint:
            best_prec1 = checkpoint['best_loss']
        else:
            best_prec1 = 0.
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        else:
            phases = ['train', 'val', 'debug']
            global_step = {x: 0 for x in phases}
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=> succeed load model '{}'".format(model_path))
        except:
            ############### TODO  Currently not compatabile to enemble network ###############
            print("Warning !!! Meet error is reading the whole model, now try to read the part")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(" The incomplelet model is succeed load from '{}'".format(model_path))
        if 'optimizer' in checkpoint:
            if not isinstance(optimizer, tuple):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                    print("=> succeed load optimizer '{}'".format(model_path))
                    optimizer.zero_grad()
                except:
                    print("Warning !!! Meet error during loading the optimize, not externaly initialized")

        return start_epoch, best_prec1, global_step
    else:
        print("=> no checkpoint found at '{}'".format(model_path))


get_test_model = resume_train


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    """
    save checkpoint during training
    'epoch': epoch,'
    :param state_dict': {'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': optimizer_state,
                  'best_score': best_score, 'global_step':global_step}
    :param is_best: if is the best model
    :param path: path to save the checkpoint
    :param prefix: prefix to add before the fname
    :param filename: filename
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    prefix_save = os.path.join(path, prefix)
    name = '_'.join([prefix_save, filename])
    try:
        torch.save(state, name,_use_new_zipfile_serialization=False)
    except:
        torch.save(state, name)

    if is_best:
        torch.save(state, path + '/model_best.pth.tar')


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w