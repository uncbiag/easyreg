import torch
import numpy as np
import skimage
import torch.nn as nn
import os
import torchvision.utils as utils
from skimage import color

def get_pair(data, pair= True, target=None):
    if pair:
        return data[:,:,0], data[:,:,1]
    else:
        return data[:,:,0], target




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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if not m.weight is None:
            nn.init.xavier_normal(m.weight.data)
        if not m.bias is None:
            nn.init.xavier_normal(m.bias.data)


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



def get_criterion(sched):
    if sched == 'L1-loss':
         sched_sel = torch.nn.L1Loss()
    elif sched == "L2-loss":
         sched_sel = torch.nn.MSELoss()
    elif sched == "W-GAN":
        raise ValueError(' not implemented')
    elif sched == 'NCC-loss':
        sched_sel = CrossCorrelationLoss
    else:
        raise ValueError(' the criterion is not implemented')
    return sched_sel


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

def resume_train(model_path, model):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
        return  start_epoch, best_prec1
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

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