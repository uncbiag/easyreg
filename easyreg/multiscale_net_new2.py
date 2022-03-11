"""
"""
import copy

from tools.visual_tools import save_3D_img_from_numpy
from .losses import NCCLoss, Loss
from .net_utils import gen_identity_map
import mermaid.finite_differences_multi_channel as fdt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import Bilinear
from .affine_net import AffineNetSym
from .utils import sigmoid_decay, get_resampled_image
import mermaid.smoother_factory as SF



class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False, residual=False, nonlinear=nn.LeakyReLU(0.2)):
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
        if residual:
            self.residual = nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=bias)
        else:
            self.residual = None


    def forward(self, x):
        x_1 = self.conv(x)
        if self.bn:
            x_1 = self.bn(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        return x_1
class FullyConnectBlock(nn.Module):
    """
    A fully connect block including fully connect layer, nonliear activiation
    """
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=nn.LeakyReLU(0.2)):
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

        super(FullyConnectBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = nonlinear

    def forward(self, x):
        x_1 = self.fc(x)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        return x_1

class LinNetwork(nn.Module):
    """
    A 2D 3D affine network

    """

    def __init__(self, img_sz=None, id_transform_pyramid=None):
        super(LinNetwork, self).__init__()
        self.img_sz = img_sz
        """ the image sz  in numpy coord"""
        self.dim = len(img_sz)
        """ the dim of image"""

        # Compute size for each layer

        self.id_transform_pyramid = id_transform_pyramid

        self.featureExtractor = nn.ModuleList()
        self.featureExtractor.append(nn.Sequential(convBlock(1, 16)))  # 160 x 160
        self.featureExtractor.append(nn.Sequential(nn.AvgPool3d(2), convBlock(16, 64)))  # 80 x 80
        self.featureExtractor.append(nn.Sequential(nn.AvgPool3d(2), convBlock(64, 64)))  # 40 x 40
        self.featureExtractor.append(nn.Sequential(nn.AvgPool3d(2), convBlock(64, 128)))  # 20 x 20
        self.featureExtractor.append(nn.Sequential(nn.AvgPool3d(2), convBlock(128, 128)))  # 10 x 10

        self.corrFeature = nn.ModuleList()
        self.corrFeature.append(nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1, groups=2)
        ))  # 200 x 200
        self.corrFeature.append(nn.Sequential(
            nn.Conv3d(128, 128, 3, padding=1, groups=2)
        ))
        self.corrFeature.append(nn.Sequential(
            nn.Conv3d(128, 128, 3, padding=1, groups=2)
        ))
        self.corrFeature.append(nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, groups=2)
        ))
        self.corrFeature.append(nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, groups=2)
        ))

        self.flowEstimator = nn.ModuleList()
        self.flowEstimator.append(nn.Sequential(
            nn.Conv3d(32 + 3, 3, 3, padding=1)
        )
        )
        self.flowEstimator.append(nn.Sequential(
            convBlock(64 + 3, 32),
            nn.Conv3d(32, 3, 3, padding=1)
        )
        )
        self.flowEstimator.append(nn.Sequential(
            # convBlock(64, 32),
            convBlock(64 + 3, 32),
            nn.Conv3d(32, 3, 3, padding=1)
        )
        )
        self.flowEstimator.append(nn.Sequential(
            # convBlock(128, 64),
            # convBlock(64, 32),
            convBlock(64 + 3, 32),
            nn.Conv3d(32, 3, 3, padding=1)
        )
        )

        self.affineEstimator = nn.Sequential(
            convBlock(64, 32),
            convBlock(32, 32, stride=2),
            nn.AdaptiveAvgPool3d(5),
            nn.Flatten(),
            FullyConnectBlock(32 * 5 * 5 * 5, 64),
            FullyConnectBlock(64, 12))

        torch.nn.init.normal_(self.affineEstimator[-1].fc.weight, mean=0., std=0.001)
        torch.nn.init.constant_(self.affineEstimator[-1].fc.bias, 0.)

        # Init flow weight with small value
        for i in range(len(self.flowEstimator)):
            torch.nn.init.normal_(self.flowEstimator[i][-1].weight, mean=0., std=0.001)
            torch.nn.init.constant_(self.flowEstimator[i][-1].bias, 0.)

        self.bilinear = Bilinear(zero_boundary=False)
        self.feature_bilinear = Bilinear(zero_boundary=True, using_scale=False)
        """ zero boundary is used for interpolated images"""

    def normalize(self,img):
        batch = img.shape[0]
        batch_min = torch.min(img.view(batch,-1), dim=1,keepdim=True)[0].view(batch,1,1,1,1)
        batch_max = torch.max(img.view(batch,-1), dim=1,keepdim=True)[0].view(batch,1,1,1,1)
        img = (img-batch_min)/(batch_max-batch_min)
        return img

    def gen_affine_map(self, Ab, shape):
        """
        generate the affine transformation map with regard to affine parameter

        :param Ab: affine parameter
        :return: affine transformation map
        """
        # Add Affine parameter to the identity transform matrix.
        Ab_cp = Ab.clone()
        Ab_cp[:, 0] += 1.
        Ab_cp[:, 5] += 1.
        Ab_cp[:, 10] += 1.

        affine_map = torch.flip(F.affine_grid(Ab_cp.view(Ab_cp.shape[0], 3, 4), shape, align_corners=True),
                                [4]).permute(0, 4, 1, 2, 3)

        # Ab = Ab.view( Ab.shape[0], 4,3) # 3d: (batch,3)
        # id_map = gen_identity_map(shape)
        # original_shape = id_map.shape
        # id_map = id_map.view(self.dim, -1)
        # affine_map = None
        # if self.dim == 3:
        #     affine_map = torch.matmul( Ab[:,:3,:], id_map)
        #     affine_map = Ab[:,3,:].contiguous().view(-1,3,1) + affine_map
        #     affine_map= affine_map.view([Ab.shape[0]] + list(original_shape))
        return affine_map

    def forward(self, moving, target, moving_seg, target_seg):
        """
        forward the affine network

        :param moving: moving image
        :param target: target image
        :return: warped image (intensity[-1,1]), transformation map (coord [-1,1]), affine param
        """
        # Parse input


        moving_cp = moving
        target_f =target
        source_f = moving
        target_f_pyramid = []
        source_f_pyramid = []
        f_idx = [0, 1, 2, 3, 4]
        for i in range(len(self.featureExtractor)):
            target_f = self.featureExtractor[i](target_f)
            source_f = self.featureExtractor[i](source_f)

            if i in f_idx:
                target_f_pyramid.append(target_f)
                source_f_pyramid.append(source_f)

        # Init the flow with affine
        corr = self.corrFeature[-1](torch.cat([target_f_pyramid[-1], source_f_pyramid[-1]], dim=1))
        affine_params = self.affineEstimator(corr[:, :int(corr.shape[1] / 2)] + corr[:, int(corr.shape[1] / 2)::])
        initial_phi = self.gen_affine_map(affine_params, source_f_pyramid[-1].shape)
        affine_map = self.gen_affine_map(affine_params,moving.shape)
        affined = self.bilinear(moving_cp, affine_map)

        flow_pyramid = [None, None, None, None, initial_phi - self.id_transform_pyramid[-1]]
        warped_from_pyramid = [None, None, None, None, affined]
        disp_pyramid = [None, None, None, None, initial_phi - self.id_transform_pyramid[-1]]
        cur_phi=None
        for i in range(len(target_f_pyramid) - 2, -1, -1):
            shape = source_f_pyramid[i].shape
            cur_disp = F.interpolate(flow_pyramid[i + 1], size=shape[2:], mode="trilinear", align_corners=True)
            source_f = self.feature_bilinear(
                source_f_pyramid[i],
                cur_disp + self.id_transform_pyramid[i]
            )

            corr = self.corrFeature[i](torch.cat([target_f_pyramid[i], source_f], dim=1))
            resid_flow = self.flowEstimator[i](
                torch.cat([corr[:, :int(corr.shape[1] / 2)] + corr[:, int(corr.shape[1] / 2)::], cur_disp], dim=1))
            flow_pyramid[i] = resid_flow + cur_disp
            cur_phi = flow_pyramid[i] + self.id_transform_pyramid[i]
            cur_phi = F.interpolate(cur_phi, size=self.img_sz, mode="trilinear", align_corners=True)
            warped_from_pyramid[i] = self.bilinear(moving_cp, cur_phi)
            disp_pyramid[i] = resid_flow

        # phi = flow_pyramid[0] + self.id_transform_pyramid[0]
        # warped = self.bilinear(moving_cp, phi)

        model_output = {"warpeds": warped_from_pyramid,
                        "phi": cur_phi,
                        "disp_pyramid": disp_pyramid,
                        "affine_params": affine_params,
                        "affine_map":affine_map,
                        "affined":affined}
        return model_output


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
        opt_multiscale_regnet = opt['tsk_set']['reg'][('multiscale_net',{},"settings for the network")]
        batch_sz = opt['tsk_set'][('batch_sz',1,"batch size ")]
        self.load_trained_affine_net = opt_multiscale_regnet[('load_trained_affine_net',False,'if true load_trained_affine_net; if false, the affine network is not initialized')]
        self.using_affine_init = opt_multiscale_regnet[("using_affine_init",True, "deploy affine network before the nonparametric network")]
        self.affine_init_path = opt_multiscale_regnet[('affine_init_path','',"the path of pretrained affine model")]
        self.affine_refine_step = opt_multiscale_regnet[('affine_refine_step', 5, "the multi-step num in affine refinement")]
        self.initial_reg_factor = opt_multiscale_regnet[('initial_reg_factor', 1., 'initial regularization factor')]
        self.min_reg_factor = opt_multiscale_regnet[('min_reg_factor', 1., 'minimum of regularization factor')]
        self.low_res_factor = opt_multiscale_regnet[('low_res_factor', 1., 'low_res_factor')]
        self.scale_sim_weight_list = opt_multiscale_regnet[("scale_weight_list",[1.0,0.8,0.6,0.4,0.2],"scale_weight_list")]
        self.scale_reg_weight_list = opt_multiscale_regnet[("scale_weight_list",[1.0,1.0,1.0,1.0,1.0],"scale_weight_list")]
        self.compute_feature_similarity = opt_multiscale_regnet[("compute_feature_similarity",False,"compute similarity in feature space")]
        self.compute_grad_image_loss = opt_multiscale_regnet[("compute_grad_image_loss",False,"compute similarity between grad image")]
        self.activate_grad_image_after_epoch = opt_multiscale_regnet[("activate_grad_image_after_epoch", 60,"activate_grad_image_after_epoch")]
        self.compute_hess_image_loss = opt_multiscale_regnet[("compute_hess_image_loss", False, "compute similarity between hess image")]
        self.activate_hess_image_after_epoch = opt_multiscale_regnet[("activate_hess_image_after_epoch", 60,"activate_hess_image_after_epoch")]
        self.activate_lncc_after_epoch = opt_multiscale_regnet[("activate_lncc_after_epoch", 100,"activate_lncc_after_epoch")]
        self.deploy_mask_during_training = opt_multiscale_regnet[("deploy_mask_during_training", False,"deploy_mask_during_training")]
        self.compute_inverse = opt_multiscale_regnet[("compute_inverse", False,"compute inverse consistency")]
        self.img_sz = img_sz
        self.spacing = 1./(np.array(img_sz)-1)
        self.double_spacing = self.spacing*2
        self.init_smoother(opt_multiscale_regnet)
        self.sim_fn = NCCLoss()
        self.extern_sim_fn = Loss(opt).criterion
        self.epoch = -1
        self.print_count = 0
        self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
        self.id_transform_pyramid = [self.id_transform.unsqueeze(0)]
        feature_sz = [img_sz]
        self.jacob_filter = Jacob()
        for i in range(4):
            feature_sz.append([int((d - 1) / 2) + 1 for d in feature_sz[i]])
            self.id_transform_pyramid.append(gen_identity_map(feature_sz[i + 1], 1.0).unsqueeze(0).cuda())
        self.network = LinNetwork(img_sz=self.img_sz, id_transform_pyramid=self.id_transform_pyramid)
        print("Attention, the affine net is not used")



    def load_pretrained_model(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")
        # cur_state = self.state_dict()
        # for key in list(checkpoint["state_dict"].keys()):
        #     if "network." in key:
        #         replaced_key = key.replace("network.", "")
        #         if replaced_key in cur_state:
        #             cur_state[replaced_key] = checkpoint["state_dict"].pop(key)
        #         else:
        #             print("")
        self.load_state_dict(checkpoint["state_dict"])
        print("load pretrained model from {}".format(pretrained_model_path))


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
        self.sim_fn = self.sim_fn if self.epoch<self.activate_lncc_after_epoch else self.extern_sim_fn
    def normalize(self,img):
        batch = img.shape[0]
        batch_min = torch.min(img.view(batch,-1), dim=1,keepdim=True)[0].view(batch,1,1,1,1)
        batch_max = torch.max(img.view(batch,-1), dim=1,keepdim=True)[0].view(batch,1,1,1,1)
        img = (img-batch_min)/(batch_max-batch_min)
        return img

    def create_avg_pyramid(self, x, levels):
        pyramids = []
        shape = x.shape[2:]
        current_x = x
        pyramids.append(x)
        for level in range(1, levels):
            shape = [int((d - 1) / 2) + 1 for d in shape]
            current_x = F.interpolate(current_x, size=shape, mode="trilinear", align_corners=True)
            pyramids.append(current_x)
        return pyramids

    def create_id_pyramid(self, shape, levels):
        pyramids = []
        for i in range(levels):
            pyramids.append(gen_identity_map(shape, 1.0).unsqueeze(0))
            shape = [int((d - 1) / 2) + 1 for d in shape]
        return pyramids

    def update_affine_reg_factor(self):
        decay_factor = 3
        factor_scale =10
        min_threshold = 0.1
        factor_scale = float(
            max(sigmoid_decay(self.epoch, static=10, k=decay_factor) * factor_scale, min_threshold))
        return factor_scale
    def network_forward(self, source, target,source_mask=None, target_mask=None,forward=False):

        # save_3D_img_from_numpy((source*source_mask)[0][0].detach().cpu().numpy(),"/playpen-raid2/zyshen/debug/source_masked_image.nii.gz")
        # save_3D_img_from_numpy(target[0][0].detach().cpu().numpy(),"/playpen-raid2/zyshen/debug/target_masked_image.nii.gz"

        model_output = self.network(source, target, source_mask, target_mask)
        disp_list, warp_list, phi = model_output["disp_pyramid"], model_output["warpeds"], model_output["phi"]

        sim_loss_list = [self.sim_fn(cur_warp, target) for cur_warp in warp_list]
        spacing_list = [torch.tensor(2 / (np.array(disp.shape[2:]) - 1), device=disp.device) for disp in disp_list]
        reg_loss_list = [self.reg_fn(disp, spacing) for disp, spacing in zip(disp_list, spacing_list)]
        sim_loss = sum([w * sim for w, sim in zip(self.scale_sim_weight_list, sim_loss_list)])
        reg_loss = sum([w * reg for w, reg in zip(self.scale_reg_weight_list, reg_loss_list)])
        affine_reg_factor = self.update_affine_reg_factor()
        affine_reg_loss = (model_output["affine_params"] ** 2).sum() * affine_reg_factor
        if self.print_count % 10 == 0 and forward:
            print("current affine reg factor is {},  loss is {}".format(affine_reg_factor, affine_reg_loss))
        reg_loss += affine_reg_loss
        return warp_list[-1], phi, model_output["affined"], sim_loss, reg_loss

    def forward(self, source, target,source_mask=None, target_mask=None):
        source_cp, target_cp = source, target
        self.update_sim_fn()
        target = self.normalize((target + 1) * target_mask) * 2 - 1
        source = self.normalize((source + 1) * source_mask) * 2 - 1
        warped, phi, affined, sim_loss, reg_loss =  self.network_forward(source, target,source_mask, target_mask,forward=True)
        if self.compute_inverse:
            inv_warped, inv_phi, inv_affined, inv_sim_loss, inv_reg_loss = self.network_forward(target, source,target_mask, source_mask)
            inverse_consistency_loss = self.compute_derivative_inverse_consistency_loss(phi, inv_phi)
            sim_loss += inv_sim_loss
            reg_loss += inverse_consistency_loss*1
            if self.print_count % 10 == 0:
                print("current inverse_consistency factor {},  loss is {}".format(1, inverse_consistency_loss))
        # if self.compute_grad_image_loss and self.epoch > self.activate_grad_image_after_epoch:
        #     sim_loss = sim_loss + self.compute_derivative_image_similarity(warp_list[-1], target, None,mode="grad")
        # if self.compute_hess_image_loss and self.epoch > self.activate_hess_image_after_epoch:
        #     sim_loss = sim_loss + self.compute_derivative_image_similarity(warp_list[-1], target, None,mode="hess")

        composed_deformed = phi
        self.sim_loss = sim_loss
        self.reg_loss = reg_loss
        self.warped = warped
        self.target = target_cp #target_list[-1]
        self.source_mask=source_mask
        self.target_mask=target_mask
        self.source  = source_cp
        if self.train:
            self.print_count += 1
        return self.warped, composed_deformed,affined

    def compute_derivative_inverse_consistency_loss(self, phi, inv_phi):
        """
        we assume phi_AB=disp_AB+identity as the forward map (source-target), phi_BA=disp_BA+identity as the inverse direction map (target-source)
        both phis are in [0,1] coordinate (to distinguish from [-1,1] coordinate)
        """
        phi_bilinear = Bilinear(zero_boundary=False, using_scale=False)
        jacob = self.jacob_filter(phi, 2. / (np.array(phi.shape[2:]) - 1))
        jacob_inv = self.jacob_filter(inv_phi, 2. / (np.array(inv_phi.shape[2:]) - 1))

        B, D, W, H, _, _ = jacob.shape
        forward = torch.linalg.norm(
            jacob * phi_bilinear(jacob_inv.view(B, D, W, H, -1).permute(0, 4, 1, 2, 3), phi).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     4,
                                                                                                                     1).view(
                B, D, W, H, 3, 3) - torch.eye(3, 3, device=jacob.device), dim=[-2, -1])
        backward = torch.linalg.norm(jacob_inv * phi_bilinear(jacob.view(B, D, W, H, -1).permute(0, 4, 1, 2, 3),
                                                              inv_phi).permute(0, 2, 3, 4, 1).view(B, D,
                                                                                                                   W, H,
                                                                                                                   3,
                                                                                                                   3) - torch.eye(
            3, 3, device=jacob.device), dim=[-2, -1])

        inverse_consistency_loss = torch.mean(forward ** 2) \
                                   + torch.mean(backward ** 2)
        return inverse_consistency_loss

    def get_extra_to_plot(self):
        return None, None

    def check_if_update_lr(self):
        return False, None

    def reg_fn(self,disp, spacing,factor=10):
        l2 = (disp)**2
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
        _, inverse_map, _ = self.forward(self.target, self.source,self.target_mask,self.source_mask)
        return inverse_map


class Jacob(torch.nn.Module):
    '''
    Compute Jacobian of 3D image using central difference.
    '''

    def __init__(self, bc_mode="linear"):
        super(Jacob, self).__init__()
        self.bc_mode = bc_mode

    def forward(self, x, spacing=(1, 1, 1)):
        '''
        :param x. Bx3xDxWxH
        '''
        g_x = F.pad(x[:, :, 2:, :, :] - x[:, :, 0:-2, :, :], (0, 0, 0, 0, 1, 1), "constant", 0) / (2 * spacing[0])
        g_y = F.pad(x[:, :, :, 2:, :] - x[:, :, :, 0:-2, :], (0, 0, 1, 1, 0, 0), "constant", 0) / (2 * spacing[1])
        g_z = F.pad(x[:, :, :, :, 2:] - x[:, :, :, :, 0:-2], (1, 1, 0, 0, 0, 0), "constant", 0) / (2 * spacing[2])

        # Use linear conditioin
        if self.bc_mode == "linear":
            g_x[:, :, 0:1, :, :] = (x[:, :, 1:2, :, :] - x[:, :, 0:1, :, :]) / spacing[0]
            g_x[:, :, -1:, :, :] = (x[:, :, -1:, :, :] - x[:, :, -2:-1, :, :]) / spacing[0]
            g_y[:, :, :, 0:1, :] = (x[:, :, :, 1:2, :] - x[:, :, :, 0:1, :]) / spacing[1]
            g_y[:, :, :, -1:, :] = (x[:, :, :, -1:, :] - x[:, :, :, -2:-1, :]) / spacing[1]
            g_z[:, :, :, :, 0:1] = (x[:, :, :, :, 1:2] - x[:, :, :, :, 0:1]) / spacing[2]
            g_z[:, :, :, :, -1:] = (x[:, :, :, :, -1:] - x[:, :, :, :, -2:-1]) / spacing[2]

        g = torch.stack([g_x.permute(0, 2, 3, 4, 1), g_y.permute(0, 2, 3, 4, 1), g_z.permute(0, 2, 3, 4, 1)], dim=-1)
        return g