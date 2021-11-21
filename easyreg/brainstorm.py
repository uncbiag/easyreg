"""
Framework described in
Data augmentation using learned transformationsfor one-shot medical image segmentation
http://www.mit.edu/~adalca/files/papers/cvpr2019_brainstorm.pdf

"""
from .net_utils import gen_identity_map
import mermaid.finite_differences as fdt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import Bilinear
import pynd.segutils as pynd_segutils


class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, max_pool=False, nonlinear=nn.LeakyReLU(0.2)):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        self.residual = residual
        self.max_pool = nn.MaxPool3d(kernel_size=(2,2,2),stride=2) if max_pool else None


    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear(x)
        if self.residual:
            x += x

        if not self.max_pool:
            return x
        else:
            y= self.max_pool(x)
            return x, y



class TransformCVPR2019(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(TransformCVPR2019, self).__init__()
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        opt_voxelmorph = opt['tsk_set']['reg']['aug_trans_net']
        self.initial_reg_factor = opt_voxelmorph[('initial_reg_factor', 1., 'initial regularization factor')]
        enc_filters = [16, 32, 32, 32 ]
        dec_filters = [32, 32, 32, 32, 32, 16, 16]
        self.enc_filter = enc_filters
        self.dec_filter = dec_filters
        input_channel =2
        output_channel= 3
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.img_sz = img_sz
        self.spacing = 1. / ( np.array(img_sz) - 1)
        self.loss_fn = None #NCCLoss()
        self.epoch = -1
        self.print_count = 0
        self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True)
        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, max_pool=True, bias=True))
            if i>0 and i<len(enc_filters)-1:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=1,max_pool=True, bias=True))
            if i ==len(enc_filters)-1:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=1,max_pool=False, bias=True))

        self.decoders.append(convBlock(enc_filters[3] + enc_filters[2],dec_filters[0], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[0] + enc_filters[1],dec_filters[1], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[1] + enc_filters[0],dec_filters[2], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[2],dec_filters[3], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[3],dec_filters[4], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[4],dec_filters[5], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[5], dec_filters[6],stride=1, bias=True))
        self.final_conv = nn.Conv3d(dec_filters[6], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.flow = nn.Conv3d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True)


    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.loss_fn = loss_fn

    def set_cur_epoch(self, cur_epoch=-1):
        """ set current epoch"""
        self.epoch = cur_epoch

    def forward(self, source, target,source_mask=None, target_mask=None):
        id_map = self.id_transform.clone()

        x_enc_0, x = self.encoders[0](torch.cat((source, target), dim=1))
        x_enc_1, x = self.encoders[1](x)
        x_enc_2, x = self.encoders[2](x)
        x_enc_3 = self.encoders[3](x)

        x = F.interpolate(x_enc_3,scale_factor=2)
        x = torch.cat((x, x_enc_2),dim=1)
        x = self.decoders[0](x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x_enc_1), dim=1)
        x = self.decoders[1](x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x_enc_0), dim=1)
        x = self.decoders[2](x)
        x = self.decoders[3](x)
        x = self.decoders[4](x)
        x = self.decoders[5](x)
        x = self.decoders[6](x)
        x = self.final_conv(x)
        disp_field = self.flow(x)

        deform_field = disp_field + id_map
        warped_source = self.bilinear(source, deform_field)
        self.warped = warped_source
        self.target = target
        self.disp_field = disp_field
        if self.train:
            self.print_count += 1
        return warped_source, deform_field, disp_field

    def get_extra_to_plot(self):
        return None, None

    def check_if_update_lr(self):
        return False, None

    def scale_reg_loss(self,sched='l2'):
        disp = self.disp_field
        fd = fdt.FD_torch(self.spacing*2)
        dfx = fd.dXc(disp[:, 0, ...])
        dfy = fd.dYc(disp[:, 1, ...])
        dfz = fd.dZc(disp[:, 2, ...])
        l2 = dfx**2+dfy**2+dfz**2
        reg = l2.mean()
        return reg


    def get_sim_loss(self):
        sim_loss = self.loss_fn.get_loss(self.warped,self.target)
        sim_loss =  sim_loss / self.warped.shape[0]
        return sim_loss

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()



    def get_loss(self):
        reg_factor =self.initial_reg_factor
        sim_loss = self.get_sim_loss()
        reg_loss = self.scale_reg_loss()
        if self.print_count % 10 == 0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(),
                                                                                                reg_loss.item(),
                                                                                                reg_factor))
        return sim_loss+ reg_factor*reg_loss


class AppearanceCVPR2019(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(AppearanceCVPR2019, self).__init__()
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        opt_voxelmorph = opt['tsk_set']['reg']['aug_appear_net']
        self.initial_reg_factor = opt_voxelmorph[('initial_reg_factor', 1., 'initial regularization factor')]
        self.sim_factor = opt_voxelmorph[('sim_factor', 1., 'initial regularization factor')]
        enc_filters = [16, 32, 32, 32, 32, 32]
        dec_filters = [64, 64, 32, 32, 32, 16, 16]
        self.enc_filter = enc_filters
        self.dec_filter = dec_filters
        input_channel =2
        output_channel= 3
        self.input_channel = 2
        self.output_channel = 3
        self.img_sz = img_sz
        self.spacing = 1. / ( np.array(img_sz) - 1)
        self.loss_fn = None #NCCLoss()
        self.epoch = -1
        self.print_count = 0
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True)
        for i in range(len(enc_filters)):
            if i == 0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, max_pool=True, bias=True))
            if i > 0 and i < len(enc_filters) - 1:
                self.encoders.append(convBlock(enc_filters[i - 1], enc_filters[i], stride=1, max_pool=True, bias=True))
            if i == len(enc_filters) - 1:
                self.encoders.append(convBlock(enc_filters[i - 1], enc_filters[i], stride=1, max_pool=False, bias=True))

        self.decoders.append(convBlock(enc_filters[5] + enc_filters[4], dec_filters[0], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[0] + enc_filters[3], dec_filters[1], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[1] + enc_filters[2], dec_filters[2], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[2] + enc_filters[1], dec_filters[3], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[3] + enc_filters[0], dec_filters[4], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[4], dec_filters[5], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[5], dec_filters[6], stride=1, bias=True))
        self.final_conv = nn.Conv3d(dec_filters[6], output_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.color = nn.Conv3d(output_channel, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.mask = None
        self.target =None
        self.reconst = None
        self.delta = None

        # identity transform for computing displacement

    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.loss_fn = loss_fn

    def set_cur_epoch(self, cur_epoch=-1):
        """ set current epoch"""
        self.epoch = cur_epoch


    def forward(self, source, target,source_mask=None, target_mask=None):

        x_enc_0,x = self.encoders[0](torch.cat((source, target), dim=1))
        x_enc_1,x = self.encoders[1](x)
        x_enc_2,x = self.encoders[2](x)
        x_enc_3,x = self.encoders[3](x)
        x_enc_4,x = self.encoders[4](x)
        x_enc_5 = self.encoders[5](x)

        x = F.interpolate(x_enc_5,scale_factor=2)
        x = torch.cat((x, x_enc_4),dim=1)
        x = self.decoders[0](x)
        x = F.interpolate(x,size=x_enc_3.shape[2:])
        x = torch.cat((x, x_enc_3), dim=1)
        x = self.decoders[1](x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x_enc_2), dim=1)
        x = self.decoders[2](x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x_enc_1), dim=1)
        x = self.decoders[3](x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, x_enc_0), dim=1)
        x = self.decoders[4](x)
        x = self.decoders[5](x)
        x = self.decoders[6](x)
        x = self.final_conv(x)

        delta = self.color(x)
        reconst = source + delta
        self.delta = delta
        self.reconst = reconst
        self.target = target
        if self.train:
            self.print_count += 1
        return reconst,delta,delta

    def get_extra_to_plot(self):
        return None, None

    def check_if_update_lr(self):
        return False, None

    def get_sim_loss(self):

        sim_loss = self.loss_fn.get_loss(self.reconst,self.target)
        sim_loss =  sim_loss / self.reconst.shape[0]
        sim_loss = sim_loss *self.sim_factor
        return sim_loss

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


    def scale_reg_loss(self):
        def __compute_contour(seg_data):
            contours = pynd_segutils.seg2contour(seg_data,exclude_zero=True, contour_type='both')[None]
            contours[contours > 0] = 1
            return torch.Tensor(contours).cuda()
        if self.mask is None:
            import SimpleITK as sitk
            atlas_path = '/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz'
            seg  = sitk.GetArrayFromImage(sitk.ReadImage(atlas_path))
            contour = __compute_contour(seg)
            self.mask = 1.0 - contour

        delta = self.delta
        fd = fdt.FD_torch(self.spacing * 2)
        dfx = fd.dXc(delta[:, 0, ...])
        dfy = fd.dYc(delta[:, 0, ...])
        dfz = fd.dZc(delta[:, 0, ...])
        dabs = dfx.abs() + dfy.abs() + dfz.abs()
        l2 = self.mask*dabs
        reg = l2.mean()
        return reg




    def get_loss(self):
        reg_factor = self.initial_reg_factor
        sim_loss = self.get_sim_loss()
        reg_loss = self.scale_reg_loss()
        if self.print_count % 10 == 0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(),
                                                                                                reg_loss.item(),
                                                                                                reg_factor))
        return sim_loss+ reg_factor*reg_loss





