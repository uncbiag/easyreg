"""
Train a 3d unet
"""
import numpy as np
import os
import sys
import gc
import shutil
import argparse
import time

import torch
# torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as utils
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from skimage import color

sys.path.append(os.path.realpath(".."))
# import datasets as data3d

import utils.transforms as bio_transform
import utils.datasets as data3d
from model import UNet3D
import utils.evalMetrics as metrics


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
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')  # logs/unet_test_checkpoint.pth.tar
    parser.add_argument('--save', default='logs', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--logdir', default='logs/patch_64_64_32_corrected', type=str, metavar='PATH',
                        help='path to tensorboard log file (default: none)')

    args = parser.parse_args()

    training_list_file = os.path.realpath("../Data/OAI_segmentation/train.txt")
    validation_list_file = os.path.realpath("../Data/OAI_segmentation/validation.txt")
    data_dir = os.path.realpath("../Data/OAI_segmentation/Nifti_cropped_rescaled")
    experiment_name = 'unet_test'

    patch_size = (64, 64, 32)  # size of 3d patch cropped from the original image (250, 165, 148)
    # patch_size = (250, 165, 148)
    n_epochs = 300
    batch_size = 4
    print_period = 4
    valid_period = 5
    n_classes = 3
    n_channels = 1

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # load training data and validation data
    print("load data")
    transform = transforms.Compose([bio_transform.RandomCrop(patch_size), bio_transform.SitkToTensor()])
    training_data = data3d.NiftiDataset(training_list_file, data_dir, "training", transform)
    training_data_loader = DataLoader(training_data, batch_size=batch_size,
                                      shuffle=True, num_workers=4)
    validation_data = data3d.NiftiDataset(validation_list_file, data_dir, "training", transform)
    validation_data_loader = DataLoader(training_data, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

    # build unet
    model = UNet3D(in_channel=n_channels, n_classes=n_classes)  # unet model
    model.cuda()

    criterion = nn.CrossEntropyLoss()  # loss function
    criterion.cuda()
    optimizer = optim.Adam(model.parameters())

    # log writer
    writer = SummaryWriter(args.logdir, experiment_name)

    # resume checkpoint or initialize
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    # training
    model.train()
    n_iters_one_epoch = len(training_data_loader.dataset)
    best_prec1 = 0
    print("Start Training")
    for epoch in range(n_epochs):
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        for i, (images, truths, name) in enumerate(training_data_loader):
            global_step = epoch * n_iters_one_epoch + i + 1  # current globel step

            # wrap inputs in Variable
            images = Variable(images).cuda()
            truths = Variable(truths.long()).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images)

            # start_time = time.time()
            # print(time.time() - start_time)

            output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(output.numel() // n_classes, n_classes)
            truths_flat = truths.view(truths.numel())
            loss = criterion(output_flat, truths_flat)
            # del output_flat, truths_flat
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]  # average loss over 10 batches
            if i % print_period == print_period - 1:  # print every 10 mini-batches
                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} ({:.3f} sec/batch)'.format
                      (epoch + 1, (i + 1) * batch_size, n_iters_one_epoch,
                       (i + 1) * batch_size / n_iters_one_epoch * 100,
                       running_loss / print_period if i > 0 else running_loss, duration / print_period))
                writer.add_scalar('loss/training', running_loss / print_period,
                                  global_step=global_step)  # data grouping by `slash`
                image_summary = make_image_summary(images, truths, output)
                writer.add_image("images/training", image_summary, global_step=global_step)
                running_loss = 0.0
                start_time = time.time()  # log running time

                # images, truths = validation_data_loader.

                # del images, truths, output, loss
                # gc.collect()

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.logdir, experiment_name)

        # validation
        if epoch % valid_period == valid_period - 1:
            model.eval()
            valid_loss = 0
            dice_FC = 0
            dice_TC = 0
            start_time = time.time()  # log running time
            for j, (images, truths, name) in enumerate(validation_data_loader):
                # wrap inputs in Variable
                images = Variable(images, volatile=True).cuda()
                truths = Variable(truths.long(), volatile=True).cuda()
                output = model(images)
                output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(output.numel() // n_classes, n_classes)
                truths_flat = truths.view(truths.numel())
                loss = criterion(output_flat, truths_flat)
                valid_loss += loss.data[0]
                pred = torch.max(output.data, 1)[1]
                dice_FC += metrics.metricEval('dice', pred.cpu().numpy() == 1, truths.cpu().data.numpy() == 1,
                                              num_labels=2)
                dice_TC += metrics.metricEval('dice', pred.cpu().numpy() == 2, truths.cpu().data.numpy() == 2,
                                              num_labels=2)
            writer.add_scalar('loss/validation', valid_loss / len(validation_data_loader),
                              global_step=global_step)  # data grouping by `slash`
            writer.add_scalars('Validation_Dice', {"FC": dice_FC / len(validation_data_loader),
                                                   "TC": dice_TC / len(validation_data_loader)},
                               global_step=global_step)
            image_summary = make_image_summary(images, truths, output)
            writer.add_image("images/validation", image_summary, global_step=global_step)

            print('Epoch: {:0d} Validation: loss: {:.3f} Dice_FC: {:.3f} Dice_TC:{:.3f} ({:.3f} sec)'.format
                  (epoch + 1, valid_loss / len(validation_data_loader), dice_FC / len(validation_data_loader),
                   dice_TC / len(validation_data_loader), time.time() - start_time))

    print('Finished Training')


if __name__ == '__main__':
    main()