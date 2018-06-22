import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


dim = 3
Conv = nn.Conv2d if dim==2 else nn.Conv3d
ConvTranspose = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
conv = F.conv2d if dim==2 else F.conv3d


class ConvBnRel(nn.Module):
    # conv + bn (optional) + relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False):
        super(ConvBnRel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = BatchNorm(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
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


class HessianField(object):
    def __init__(self):
        if dim ==2:
            self.laplace = Variable(torch.cuda.FloatTensor([0.,1.,0.],[1.,-4,1.],[0.,1.,0.])).view(1,1,3,3)
        elif dim==3:
            self.laplace = Variable(torch.cuda.FloatTensor([[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                                            [[0., 1., 0.], [1., -6, 1.], [0., 1., 0.]]
                                                            [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]
                                                           )).view(1,1,3,3,3)
    def __call__(self,disField):
        x = conv(disField,self.laplace)
        return x



class HessianField(object):
    def __init__(self):
        if dim ==2:
            dx = Variable(torch.cuda.FloatTensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])).view(1,1,3,3)
            dy = Variable(torch.cuda.FloatTensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])).view(1,1,3,3)
            self.spatial_filter = torch.cat((dx,dy),1)
            self.spatial_filter.repeat(2, 1, 1, 1)
        elif dim==3:
            dx = Variable(torch.cuda.FloatTensor([[[-1., -3., -1.], [-3., -6., -3.], [-1., -3., -1.]],
                                                            [[0., 0., 0.], [0., 0, 0.], [0., 0., 0.]],
                                                            [[1., 3., 1.], [3., 6., 3.], [1., 3., 1.]]]
                                                           )).view(1,1,3,3,3)
            dy  = Variable(
                torch.cuda.FloatTensor([[[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]],
                                        [[3., 6., 3.], [0., 0, 0.], [-3., -6., -3.]],
                                        [[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]]]
                                       )).view(1, 1, 3, 3, 3)
            dz = Variable(
                torch.cuda.FloatTensor([[[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]],
                                        [[-3., 0., 3.], [-6., 0, 6.], [-3., 0., 3.]],
                                        [[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]]]
                                       )).view(1, 1, 3, 3, 3)
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
