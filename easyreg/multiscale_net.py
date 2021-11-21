"""
"""
import copy
from .losses import NCCLoss, LNCCLoss
from .net_utils import gen_identity_map
import mermaid.finite_differences_multi_channel as fdt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import Bilinear
from mermaid.libraries.modules import stn_nd
from .affine_net import AffineNetSym
from .utils import sigmoid_decay, get_resampled_image
import mermaid.utils as py_utils
import mermaid.smoother_factory as SF


class conv_bn_rel(nn.Module):
    """
    conv + bn (optional) + relu

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, group=1, dilation=1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation)

        self.bn = nn.BatchNorm3d(out_channels) if bn else None #, eps=0.0001, momentum=0, affine=True
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

class Multiscale_Flow(nn.Module):
    def __init__(self, img_sz, low_res_factor=1, batch_sz=1, compute_feature_similarity=False, bn=False):
        super(Multiscale_Flow,self).__init__()
        self.img_sz = img_sz
        self.low_res_factor = low_res_factor
        self.compute_feature_similarity = compute_feature_similarity # not support yet in this framework
        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_1 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False,group=2)
        self.down_path_2_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_4_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_4_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_8_1 = conv_bn_rel(64, 128, 3, stride=2, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_8_2 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn,group=2)
        self.down_path_16_1 = conv_bn_rel(128, 256, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.flow_conv_16 = conv_bn_rel(256, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)
        self.upsample_16_8 = Bilinear(zero_boundary=False,using_scale=False)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(256, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_8_2= conv_bn_rel(128+128+3, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_8_3= conv_bn_rel(128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.flow_conv_8  = conv_bn_rel(128, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)
        img_sz_8 = [batch_sz,1] +list([int(d/8) for d in self.img_sz])
        spacing_8 = 1/(np.array(img_sz_8[2:])-1)
        id_map_8 = py_utils.identity_map_multiN(img_sz_8, spacing_8)*2-1
        self.id_map_8 = torch.Tensor(id_map_8).cuda()

        self.interp_8 = Bilinear(zero_boundary=False,using_scale=False)
        self.sinterp_8 =  Bilinear(zero_boundary=True,using_scale=True)
        self.tinterp_8 =  Bilinear(zero_boundary=True,using_scale=True)
        self.upsample_8_4 = Bilinear(zero_boundary=False,using_scale=False)

        self.up_path_4_1 = conv_bn_rel(128, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_4_2 = conv_bn_rel(64+64+3, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.flow_conv_4  = conv_bn_rel(32, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)
        img_sz_4 = [batch_sz, 1] + list([int(d/4) for d in self.img_sz])
        spacing_4 = 1 / (np.array(img_sz_4[2:]) - 1)
        id_map_4 = py_utils.identity_map_multiN(img_sz_4, spacing_4)*2-1
        self.id_map_4 = torch.Tensor(id_map_4).cuda()
        self.interp_4 = Bilinear(zero_boundary=False,using_scale=False)
        self.sinterp_4 = Bilinear(zero_boundary=True, using_scale=True)
        self.tinterp_4 = Bilinear(zero_boundary=True, using_scale=True)
        self.upsample_4_2 = Bilinear(zero_boundary=False,using_scale=False)

        self.up_path_2_1 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,reverse=True)
        self.up_path_2_2 = conv_bn_rel(32+32+3, 16, 3, stride=1, active_unit='None', same_padding=True)
        self.up_path_2_3 = conv_bn_rel(16, 16, 3, stride=1, active_unit='None', same_padding=True)
        self.flow_conv_2 = conv_bn_rel(16, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)
        img_sz_2 = [batch_sz, 1] + list([int(d/2) for d in self.img_sz])
        spacing_2 = 1 / (np.array(img_sz_2[2:]) - 1)
        id_map_2 = py_utils.identity_map_multiN(img_sz_2, spacing_2) * 2 - 1
        self.id_map_2 = torch.Tensor(id_map_2).cuda()
        self.interp_2 = Bilinear(zero_boundary=False,using_scale=False)
        self.sinterp_2 = Bilinear(zero_boundary=True, using_scale=True)
        self.tinterp_2 = Bilinear(zero_boundary=True, using_scale=True)
        self.upsample_2_1 = Bilinear(zero_boundary=False,using_scale=False)

        img_sz_1 = [batch_sz, 1] + list([int(d/1) for d in self.img_sz])
        spacing_1 = 1 / (np.array(img_sz_1[2:]) - 1)
        id_map_1 = py_utils.identity_map_multiN(img_sz_1, spacing_1) * 2 - 1
        self.id_map_1 = torch.Tensor(id_map_1).cuda()
        self.interp_1 = Bilinear(zero_boundary=False,using_scale=False)
        self.sinterp_1 = Bilinear(zero_boundary=True, using_scale=True)
        self.tinterp_1 = Bilinear(zero_boundary=True, using_scale=True)
        self.minterp_1 = Bilinear(zero_boundary=False, using_scale=False)

        if self.low_res_factor==1:
            self.up_path_1_1 = conv_bn_rel(16, 16, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn,
                                           reverse=True)
            self.up_path_1_2 = conv_bn_rel(16+16+3, 8, 3, stride=1, active_unit='None', same_padding=True)
            self.up_path_1_3 = conv_bn_rel(8, 8, 3, stride=1, active_unit='None', same_padding=True)
            self.flow_conv_1 = conv_bn_rel(8, 3, 3, stride=1, active_unit='None', same_padding=True, bn=False)

    def forward(self, source,target, initial_map,smoother=None):
        input_cat = torch.cat((source, target), dim=1)
        d1 = self.down_path_1(input_cat)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2+ d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2
        flow_16 = self.flow_conv_16(d16_2)
        flow_16_8 = self.upsample_16_8(flow_16,self.id_map_8)
        deform_field_8 = self.id_map_8 + flow_16_8
        warped_8 = self.sinterp_8(source, deform_field_8)
        target_8 = self.tinterp_8(target, self.id_map_8)


        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3,u8_1,flow_16_8),1))
        u8_3 = self.up_path_8_3(u8_2)
        flow_8 = self.flow_conv_8(u8_3)+ flow_16_8
        flow_8_4 = self.upsample_8_4(flow_8,self.id_map_4)
        deform_field_4 = self.id_map_4 + flow_8_4
        warped_4 = self.sinterp_4(source, deform_field_4)
        target_4 = self.tinterp_4(target, self.id_map_4)


        u4_1 = self.up_path_4_1(u8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3, u4_1, flow_8_4), 1))
        u4_3 = self.up_path_4_3(u4_2)
        flow_4 = self.flow_conv_4(u4_3) +flow_8_4
        flow_4_2 = self.upsample_4_2(flow_4, self.id_map_2)
        deform_field_2 = self.id_map_2 + flow_4_2
        warped_2 = self.sinterp_2(source, deform_field_2)
        target_2 = self.tinterp_2(target, self.id_map_2)


        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1, flow_4_2), 1))
        u2_3 = self.up_path_2_3(u2_2)
        flow_2 = self.flow_conv_2(u2_3) + flow_4_2
        flow_2_1 = self.upsample_2_1(flow_2, self.id_map_1)

        if self.low_res_factor==1:
            u1_1 = self.up_path_1_1(u2_3)
            u1_2 = self.up_path_1_2(torch.cat((d1, u1_1, flow_2_1), 1))
            u1_3 = self.up_path_1_3(u1_2)
            flow_1 = self.flow_conv_1(u1_3) + flow_2_1
            sm_flow_1 = flow_1 if smoother is None else smoother.smooth(flow_1)
            deform_field_1 = self.id_map_1 + sm_flow_1
        else:
            flow_1 = flow_2_1
            sm_flow_1 = flow_2_1 if smoother is None else smoother.smooth(flow_2_1)
            deform_field_1 = self.id_map_1 + sm_flow_1

        warped_1 = self.sinterp_1(source, deform_field_1)
        target_1 = target
        deformed_map =self.minterp_1(initial_map,deform_field_1)
        flow_list = [flow_8,flow_4-flow_8_4,flow_2-flow_4_2,flow_1-flow_2_1]
        warp_list = [warped_8,warped_4,warped_2,warped_1]
        target_list = [target_8, target_4,target_2, target_1]
        return flow_list, warp_list, target_list, deformed_map


class Multiscale_FlowNet(nn.Module):
    """
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(Multiscale_FlowNet, self).__init__()
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        opt_multiscale_regnet = opt['tsk_set']['reg']['multiscale_net']
        batch_sz = opt['tsk_set']['batch_sz']
        self.load_trained_affine_net = opt_multiscale_regnet[('load_trained_affine_net',False,'if true load_trained_affine_net; if false, the affine network is not initialized')]
        self.using_affine_init = opt_multiscale_regnet[("using_affine_init",False, "deploy affine network before the nonparametric network")]
        self.affine_init_path = opt_multiscale_regnet[('affine_init_path','',"the path of pretrained affine model")]
        self.affine_refine_step = opt_multiscale_regnet[('affine_refine_step', 5, "the multi-step num in affine refinement")]
        self.initial_reg_factor = opt_multiscale_regnet[('initial_reg_factor', 1., 'initial regularization factor')]
        self.min_reg_factor = opt_multiscale_regnet[('min_reg_factor', 1., 'minimum of regularization factor')]
        self.low_res_factor = opt_multiscale_regnet[('low_res_factor', 1., 'low_res_factor')]
        self.scale_weight_list = opt_multiscale_regnet[("scale_weight_list",[1.0,0.8,0.6,0.4],"scale_weight_list")]
        self.compute_feature_similarity = opt_multiscale_regnet[("compute_feature_similarity",False,"compute similarity in feature space")]
        self.compute_grad_image_loss = opt_multiscale_regnet[("compute_grad_image_loss",False,"compute similarity between grad image")]
        self.activate_grad_image_after_epoch = opt_multiscale_regnet[("activate_grad_image_after_epoch", 60,"activate_grad_image_after_epoch")]
        self.compute_hess_image_loss = opt_multiscale_regnet[("compute_hess_image_loss", False, "compute similarity between hess image")]
        self.activate_hess_image_after_epoch = opt_multiscale_regnet[("activate_hess_image_after_epoch", 60,"activate_hess_image_after_epoch")]
        self.activate_lncc_after_epoch = opt_multiscale_regnet[("activate_lncc_after_epoch", 100,"activate_lncc_after_epoch")]
        self.deploy_mask_during_training = opt_multiscale_regnet[("deploy_mask_during_training", False,"deploy_mask_during_training")]
        self.img_sz = img_sz
        self.spacing = 1./(np.array(img_sz)-1)
        self.double_spacing = self.spacing*2
        self.network = Multiscale_Flow(img_sz=self.img_sz, low_res_factor=self.low_res_factor,batch_sz=batch_sz)
        self.init_smoother(opt_multiscale_regnet)
        self.input_channel = 2
        self.output_channel = 3
        self.sim_fn = NCCLoss()
        self.epoch = -1
        self.print_count = 0



        if self.using_affine_init:
            self.init_affine_net(opt)
            self.id_transform = None
        else:
            self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
            print("Attention, the affine net is not used")



        # identity transform for computing displacement


    def init_smoother(self, opt):
        #the output displacement is defined on the transformation map [-1,1]
        self.flow_smoother = SF.SmootherFactory(self.img_sz, self.double_spacing).create_smoother(opt)
        opt_cp= copy.deepcopy(opt)
        opt_cp["smoother"]["type"] = "gaussian"
        opt_cp["smoother"]["gaussian_std"] = 0.1
        self.mask_smoother = SF.SmootherFactory(self.img_sz, self.double_spacing).create_smoother(opt_cp)


    def set_cur_epoch(self, cur_epoch=-1):
        """ set current epoch"""
        self.epoch = cur_epoch


    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.loss_fn = loss_fn

    def init_affine_net(self,opt):
        self.affine_net = AffineNetSym(self.img_sz, opt)
        self.affine_net.compute_loss = False
        self.affine_net.epoch_activate_sym = 1e7  # todo to fix this unatural setting
        self.affine_net.set_step(self.affine_refine_step)
        model_path = self.affine_init_path
        if self.load_trained_affine_net and self.is_train:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.affine_net.load_state_dict(checkpoint['state_dict'])
            self.affine_net.cuda()
            print("Affine model is initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print(
                "The Affine model is added, but not initialized, this should only take place when a complete checkpoint (including affine model) will be loaded")
        self.affine_net.eval()


    def compute_grad_image(self, image):
        fd = fdt.FD_torch_multi_channel(self.spacing)
        dfx = fd.dXf(image)
        dfy = fd.dYf(image)
        dfz = fd.dZf(image)
        grad_image = torch.cat([dfx,dfy,dfz],1)
        return grad_image

    def compute_hessian_image(self,image):
        fd = fdt.FD_torch_multi_channel(self.spacing)
        ddfx = fd.ddXc(image)
        ddfy = fd.ddYc(image)
        ddfz = fd.ddZc(image)
        hess_image = torch.cat([ddfx,ddfy,ddfz],1)
        return hess_image


    def compute_derivative_image_similarity(self,warp, target, target_mask=None,mode="grad"):
        compute_grad = True if mode=="grad" else False
        derivative_fn = self.compute_grad_image if compute_grad else self.compute_hessian_image
        grad_warp = derivative_fn(warp)
        grad_target = derivative_fn(target)
        if target_mask is not None:
            target_mask = self.mask_smoother.smooth(target_mask)
            target_mask = target_mask.repeat([1,3,1,1,1])
        sim_loss = NCCLoss()(grad_warp, grad_target, mask=target_mask)
        return sim_loss

    def update_sim_fn(self):
        self.sim_fn = self.sim_fn if self.epoch<self.activate_lncc_after_epoch else LNCCLoss()

    def forward(self, source, target,source_mask=None, target_mask=None):
        self.update_sim_fn()
        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, _ = self.affine_net(source, target)
        else:
            affine_map = self.id_transform.clone()
            affine_img = source

        disp_list, warp_list, target_list, phi = self.network(affine_img, target, affine_map, self.flow_smoother)
        sim_loss_list = [self.sim_fn(cur_warp, cur_targ) for cur_warp, cur_targ in zip(warp_list, target_list)]
        spacing_list = [2/(np.array(disp.shape[2:])-1) for disp in disp_list]
        reg_loss_list = [self.reg_fn(disp,spacing) for disp, spacing in zip(disp_list, spacing_list)]
        sim_loss = sum([w*sim for w, sim in zip(self.scale_weight_list, sim_loss_list)])
        reg_loss = sum([w*reg for w, reg in zip(self.scale_weight_list, reg_loss_list)])
        if self.compute_grad_image_loss and self.epoch > self.activate_grad_image_after_epoch:
            sim_loss = sim_loss + self.compute_derivative_image_similarity(warp_list[-1], target, target_mask,mode="grad")
        if self.compute_hess_image_loss and self.epoch > self.activate_hess_image_after_epoch:
            sim_loss = sim_loss + self.compute_derivative_image_similarity(warp_list[-1], target, target_mask,mode="hess")
        composed_deformed = phi
        self.sim_loss = sim_loss
        self.reg_loss = reg_loss
        self.warped = warp_list[-1]
        self.target = target_list[-1]
        self.disp_field = disp_list[-1]
        self.source  = source
        if self.train:
            self.print_count += 1
        return self.warped, composed_deformed, self.disp_field

    def get_extra_to_plot(self):
        return None, None

    def check_if_update_lr(self):
        return False, None

    def reg_fn(self,disp, spacing):
        l2 = disp**2
        reg = l2.mean()
        return reg


    def get_sim_loss(self):
        return self.sim_loss

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


    def get_reg_factor(self):
        factor = self.initial_reg_factor # 1e-7
        factor = float(max(sigmoid_decay(self.epoch, static=5, k=4) * factor, self.min_reg_factor))
        return factor


    def get_loss(self):
        reg_factor = self.get_reg_factor()
        sim_loss = self.sim_loss
        reg_loss = self.reg_loss
        if self.print_count % 10 == 0:
            print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(),
                                                                                                reg_loss.item(),
                                                                                                reg_factor))
        return sim_loss+ reg_factor*reg_loss

    def get_inverse_map(self, use_01=False):
        # TODO  not test yet
        print("VoxelMorph approach doesn't support analytical computation of inverse map")
        print("Instead, we compute it's numerical approximation")
        _, inverse_map, _ = self.forward(self.target, self.source)
        return inverse_map

