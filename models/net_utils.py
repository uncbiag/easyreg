import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


dim = 3
Conv = nn.Conv2d if dim==2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim ==2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
conv = F.conv2d if dim==2 else F.conv3d


class conv_bn_rel(nn.Module):
    # conv + bn (optional) + relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False,group = 1,dilation = 1,padding_num=None):
        super(conv_bn_rel, self).__init__()
        if padding_num is None:
            padding = int((kernel_size - 1) / 2) if same_padding else 0
        else:
            padding = padding_num
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1,dilation=1)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding,groups=1,dilation=1)

        self.bn = BatchNorm(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit =='leaky_relu':
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
    # fc+ relu(option)
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




def identity_map(sz):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[-1:1.:2./sz[0]]
    elif dim == 2:
        id = np.mgrid[-1.:1.:2./sz[0], -1.:1.:2./sz[1]]
    elif dim == 3:
        #id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[-1.:1.:2./sz[0],-1.:1.:2./sz[1],-1.:1.:2./sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')
    #id= id*2-1
    return Variable(torch.from_numpy(id.astype(np.float32)).cuda())



def gen_identity_map(img_sz,resize_factor=1.):
    """
    given displacement field,  add displacement on grid field
    """
    if isinstance(resize_factor,list):
        img_sz = [int(img_sz[i]*resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i]*resize_factor) for i in range(dim)]
    grid = identity_map(img_sz)
    return grid



# class HessianField(object):
#     def __init__(self):
#         if dim ==2:
#             self.laplace = Variable(torch.cuda.FloatTensor([0.,1.,0.],[1.,-4,1.],[0.,1.,0.])).view(1,1,3,3)
#         elif dim==3:
#             self.laplace = Variable(torch.cuda.FloatTensor([[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
#                                                             [[0., 1., 0.], [1., -6, 1.], [0., 1., 0.]]
#                                                             [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]
#                                                            )).view(1,1,3,3,3)
#     def __call__(self,disField):
#         x = conv(disField,self.laplace)
#         return x



class HessianField(object):
    def __init__(self):
        if dim ==2:
            dx = Variable(torch.cuda.FloatTensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])).view(1,1,3,3)
            dy = Variable(torch.cuda.FloatTensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])).view(1,1,3,3)
            self.spatial_filter = torch.cat((dx,dy),1)
            self.spatial_filter.v(2, 1, 1, 1)
        elif dim==3:
            dx = Variable(torch.cuda.FloatTensor([[[-1., -3., -1.], [-3., -6., -3.], [-1., -3., -1.]],
                                                            [[0., 0., 0.], [0., 0, 0.], [0., 0., 0.]],
                                                            [[1., 3., 1.], [3., 6., 3.], [1., 3., 1.]]]
                                                           )).view(1,1,3,3,3)/10
            dy  = Variable(
                torch.cuda.FloatTensor([[[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]],
                                        [[3., 6., 3.], [0., 0, 0.], [-3., -6., -3.]],
                                        [[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]]]
                                       )).view(1, 1, 3, 3, 3)/10
            dz = Variable(
                torch.cuda.FloatTensor([[[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]],
                                        [[-3., 0., 3.], [-6., 0, 6.], [-3., 0., 3.]],
                                        [[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]]]
                                       )).view(1, 1, 3, 3, 3)/10
            self.spatial_filter = torch.cat((dx, dy,dz), 1)
            self.spatial_filter=self.spatial_filter.repeat(3,1,1,1,1)

    def __call__(self,disField):
        hessionField = conv(disField, self.spatial_filter)
        if dim==2:
            return hessionField[:, 0:1, ...] ** 2 + hessionField[:, 1:2, ...] ** 2

        elif dim==3:
            return hessionField[:, 0:1, ...] ** 2 + hessionField[:, 1:2, ...] ** 2 + hessionField[:, 2:3, ...] ** 2


class JacobiField(object):
    def __call__(self,disField):
        if dim==2:
            return disField[:,0:1,...]**2 + disField[:,1:2,...]**2
        elif dim==3:
            return disField[:,0:1,...]**2 + disField[:,1:2,...]**2 +disField[:,2:3,...]**2


class AffineConstrain(object):
    def __init__(self):
        if dim==3:
            self.affine_identity = Variable(torch.zeros(12)).cuda()
            self.affine_identity[0] = 1.
            self.affine_identity[4] = 1.
            self.affine_identity[8] = 1.
        else:
            raise ValueError("Not Implemented")
    def __call__(self,affine_param, sched='l2'):
        if sched=='l2':
            return (self.affine_identity-affine_param)**2
        elif sched=='det':
            mean_det = 0.
            for i in range(affine_param.shape[0]):
                affine_matrix = affine_param[i,:9].contiguous().view(3,3)
                mean_det += torch.det(affine_matrix)
            return  mean_det / affine_param.shape[0]


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        # like the weight and bias
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad




def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


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
    if isinstance(tensors, Variable):
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
            centered_id = centered_identity_map(sz,spacing)
            g = compute_normalized_gaussian(centered_id, mus, stds)
            tensors[n,c] = torch.from_numpy(g)




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