"""
registration network described in voxelmorph
An experimental pytorch implemetation, the official tensorflow please refers to https://github.com/voxelmorph/voxelmorph

An Unsupervised Learning Model for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
CVPR 2018. eprint arXiv:1802.02604

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605
"""
from .net_utils import gen_identity_map
import mermaid.finite_differences as fdt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import Bilinear
from mermaid.libraries.modules import stn_nd
from .affine_net import AffineNetSym
from .utils import sigmoid_decay

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
        self.residual = residual

    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear(x)
        if self.residual:
            x += x

        return x


class VoxelMorphCVPR2018(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(VoxelMorphCVPR2018, self).__init__()
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        opt_voxelmorph = opt['tsk_set']['reg']['vm_cvpr']
        self.load_trained_affine_net = opt_voxelmorph[('load_trained_affine_net',False,'if true load_trained_affine_net; if false, the affine network is not initialized')]
        self.using_affine_init = opt_voxelmorph[("using_affine_init",False, "deploy affine network before the nonparametric network")]
        self.affine_init_path = opt_voxelmorph[('affine_init_path','',"the path of pretrained affine model")]
        self.affine_refine_step = opt_voxelmorph[('affine_refine_step', 5, "the multi-step num in affine refinement")]
        self.initial_reg_factor = opt_voxelmorph[('initial_reg_factor', 1., 'initial regularization factor')]
        self.min_reg_factor = opt_voxelmorph[('min_reg_factor', 1., 'minimum of regularization factor')]
        enc_filters = [16, 32, 32, 32, 32]
        #dec_filters = [32, 32, 32, 8, 8]
        dec_filters = [32, 32, 32, 32, 32, 16, 16]
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

        def set_cur_epoch(self, cur_epoch=-1):
            """ set current epoch"""
            self.epoch = cur_epoch

        if self.using_affine_init:
            self.init_affine_net(opt)
            self.id_transform = None
        else:
            self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
            print("Attention, the affine net is not used")


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bilinear = Bilinear(zero_boundary=True)
        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        self.decoders.append(convBlock(enc_filters[-1], dec_filters[0], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[0] + enc_filters[3],dec_filters[1], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[1] + enc_filters[2],dec_filters[2], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[2] + enc_filters[1],dec_filters[3], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[3], dec_filters[4],stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[4] + enc_filters[0],dec_filters[5], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[5], dec_filters[6],stride=1, bias=True))


        self.flow = nn.Conv3d(dec_filters[-1], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        # identity transform for computing displacement

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

    def forward(self, source, target,source_mask=None, target_mask=None):

        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, _ = self.affine_net(source, target)
        else:
            affine_map = self.id_transform.clone()
            affine_img = source

        x_enc_1 = self.encoders[0](torch.cat((affine_img, target), dim=1))
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        x = self.decoders[0](x_enc_5)
        x = F.interpolate(x,scale_factor=2,mode='trilinear')
        x = torch.cat((x, x_enc_4),dim=1)
        x = self.decoders[1](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat((x, x_enc_3), dim=1)
        x = self.decoders[2](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat((x, x_enc_2), dim=1)
        x = self.decoders[3](x)
        x = self.decoders[4](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat((x, x_enc_1), dim=1)
        x = self.decoders[5](x)
        x = self.decoders[6](x)

        disp_field = self.flow(x)
        #del x_dec_5, x_enc_1

        deform_field = disp_field + affine_map
        warped_source = self.bilinear(source, deform_field)
        self.warped = warped_source
        self.target = target
        self.disp_field = disp_field
        self.source  = source
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
        sim_loss = sim_loss / self.warped.shape[0]
        return sim_loss

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
        sim_loss = self.get_sim_loss()
        reg_loss = self.scale_reg_loss()
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

    # def cal_affine_loss(self,output=None,afimg_or_afparam=None,using_decay_factor=False):
    #     factor = 1.0
    #     if using_decay_factor:
    #         factor = sigmoid_decay(self.cur_epoch,static=5, k=4)*factor
    #     if self.loss_fn.criterion is not None:
    #         sim_loss  = self.loss_fn.get_loss(output,self.target)
    #     else:
    #         sim_loss = self.network.get_sim_loss(output,self.target)
    #     reg_loss = self.network.scale_reg_loss(afimg_or_afparam) if afimg_or_afparam is not None else 0.
    #     if self.iter_count%10==0:
    #         print('current sim loss is{}, current_reg_loss is {}, and reg_factor is {} '.format(sim_loss.item(), reg_loss.item(),factor))
    #     return sim_loss+reg_loss*factor



class VoxelMorphMICCAI2019(nn.Module):
    """
    unet architecture for voxelmorph models presented in the MICCAI 2019 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the reg_model
    """
    def __init__(self, img_sz, opt=None):
        super(VoxelMorphMICCAI2019, self).__init__()
        self.is_train = opt['tsk_set'][('train', False, 'if is in train mode')]
        opt_voxelmorph = opt['tsk_set']['reg']['vm_miccai']
        self.load_trained_affine_net = opt_voxelmorph[('load_trained_affine_net', False,
                                                       'if true load_trained_affine_net; if false, the affine network is not initialized')]
        self.affine_refine_step = opt_voxelmorph[('affine_refine_step', 5, "the multi-step num in affine refinement")]

        self.using_affine_init = opt_voxelmorph[("using_affine_init", False,  "deploy affine network before the nonparametric network")]
        self.affine_init_path = opt_voxelmorph[('affine_init_path', '', "the path of pretrained affine model")]
        enc_filters = [16, 32, 32, 32, 32]
        #dec_filters = [32, 32, 32, 8, 8]
        dec_filters = [32, 32, 32, 32, 16]
        self.enc_filter = enc_filters
        self.dec_filter = dec_filters
        input_channel =2
        output_channel= 3
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.img_sz = img_sz
        self.low_res_img_sz = [int(x/2) for x in img_sz]
        self.spacing = 1. / ( np.array(img_sz) - 1)
        self.int_steps = 7

        self.image_sigma = opt_voxelmorph[('image_sigma',0.02,'image_sigma')]
        self.prior_lambda =opt_voxelmorph[('lambda_factor_in_vmr',50,'lambda_factor_in_vmr')]
        self.prior_lambda_mean =opt_voxelmorph[('lambda_mean_factor_in_vmr',50,'lambda_mean_factor_in_vmr')]
        self.flow_vol_shape = self.low_res_img_sz
        self.D = self._degree_matrix(self.flow_vol_shape)
        self.D = (self.D).cuda()# 1, 96, 40,40 3'
        self.loss_fn =  None


        if self.using_affine_init:
            self.init_affine_net(opt)
            self.id_transform = None
        else:
            self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda()
            self.id_transform  =self.id_transform.view([1]+list(self.id_transform.shape))
            print("Attention, the affine net is not used")
        """to compatiable to the mesh setting in voxel morph"""
        self.low_res_id_transform = gen_identity_map(self.img_sz, 0.5, normalized=False).cuda()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        #self.bilinear = Bilinear(zero_boundary=True)
        self.bilinear = stn_nd.STN_ND_BCXYZ(np.array([1.,1.,1.]),zero_boundary=True)
        self.bilinear_img = Bilinear(zero_boundary=True)
        for i in range(len(enc_filters)):
            if i==0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        self.decoders.append(convBlock(enc_filters[-1], dec_filters[0], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[0] + enc_filters[3],dec_filters[1], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[1] + enc_filters[2],dec_filters[2], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[2] + enc_filters[1],dec_filters[3], stride=1, bias=True))
        self.decoders.append(convBlock(dec_filters[3], dec_filters[4],stride=1, bias=True))

        self.flow_mean =  nn.Conv3d(dec_filters[-1], output_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_sigma =  nn.Conv3d(dec_filters[-1], output_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_mean.weight.data.normal_(0.,1e-5)
        self.flow_sigma.weight.data.normal_(0.,1e-10)
        self.flow_sigma.bias.data = torch.Tensor([-10]*3)
        self.print_count=0
        # identity transform for computing displacement

    def scale_map(self,map, spacing):
        """
        Scales the map to the [-1,1]^d format

        :param map: map in BxCxXxYxZ format
        :param spacing: spacing in XxYxZ format
        :return: returns the scaled map
        """
        sz = map.size()
        map_scaled = torch.zeros_like(map)
        ndim = len(spacing)

        # This is to compensate to get back to the [-1,1] mapping of the following form
        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

        for d in range(ndim):
            if sz[d + 2] > 1:
                map_scaled[:, d, ...] = map[:, d, ...] * (2. / (sz[d + 2] - 1.) / spacing[d])
            else:
                map_scaled[:, d, ...] = map[:, d, ...]

        return map_scaled

    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        pass

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

    def forward(self, source, target,source_mask=None, target_mask=None):
        self.__do_some_clean()

        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, _ = self.affine_net(source, target)
        else:
            affine_map = self.id_transform.clone()
            affine_img = source

        x_enc_1 = self.encoders[0](torch.cat((affine_img, target), dim=1))
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        x = self.decoders[0](x_enc_5)
        x = F.interpolate(x,scale_factor=2,mode='trilinear')
        x = torch.cat((x, x_enc_4),dim=1)
        x = self.decoders[1](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat((x, x_enc_3), dim=1)
        x = self.decoders[2](x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat((x, x_enc_2), dim=1)
        x = self.decoders[3](x)
        x = self.decoders[4](x)
        flow_mean = self.flow_mean(x)
        log_sigma = self.flow_sigma(x)
        noise = torch.randn(flow_mean.shape).cuda()
        flow = flow_mean + torch.exp(log_sigma / 2.0) * noise
        #print("the min and max of flow_mean is {} {}, of the flow is {},{} ".format(flow_mean.min(),flow_mean.max(),flow.min(), flow.max()))

        for _ in range(self.int_steps):
            deform_field = flow + self.low_res_id_transform
            flow_1 = self.bilinear(flow, deform_field)
            flow = flow_1+ flow
        #print("the min and max of  self.low_res_id_transform after is {} {}".format( self.low_res_id_transform.min(),  self.low_res_id_transform.max()))
        #print("the min and max of flow after is {} {}".format(flow.min(),flow.max()))



        disp_field = F.interpolate(flow, scale_factor=2, mode='trilinear')
        disp_field=self.scale_map(disp_field,np.array([1,1,1]))
        deform_field = disp_field + affine_map
        #print("the min and max of disp_filed is {} {}, of the deform field is {},{} ".format(disp_field.min(),disp_field.max(),deform_field.min(), deform_field.max()))
        warped_source = self.bilinear_img(source, deform_field)
        self.afimg_or_afparam = disp_field
        self.res_flow_mean  =  flow #flow_mean TODO  in original code here is flow_mean, but it doesn't work
        self.res_log_sigma = log_sigma
        self.warped = warped_source
        self.target = target
        self.source = source
        if self.train:
            self.print_count += 1

        return warped_source, deform_field, disp_field


    def check_if_update_lr(self):
        return False, None

    def get_extra_to_plot(self):
        return None, None
    def __do_some_clean(self):
        self.afimg_or_afparam = None
        self.res_flow_mean = None
        self.res_log_sigma = None
        self.warped = None
        self.target = None
        self.source = None

    def scale_reg_loss(self,disp=None,sched='l2'):
        reg = self.kl_loss()
        return reg

    def get_sim_loss(self, warped=None, target=None):
        loss = self.recon_loss()
        return loss


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)  # 3 3 3
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        filt = np.zeros([ndims, ndims]+ [3] * ndims )  # 3 3 3 3  ##!!!!!!!! in out w h d
        for i in range(ndims):
            filt[ i, i,...] = filt_inner  ##!!!!!!!!

        return filt

    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [ndims,*vol_shape]  # 96 96 40 3  ##!!!!!!!!

        # prepare conv kernel
        conv_fn = F.conv3d  ##!!!!!!!!

        # prepare tf filter
        z = torch.ones([1] + sz)  # 1 96 96 40 3
        filt_tf = torch.Tensor(self._adj_filt(ndims))  # 3 3 3 3 ##!!!!!!!!
        strides = [1] * (ndims)  ##!!!!!!!!
        return conv_fn(z, filt_tf, padding= 1, stride =strides)  ##!!!!!!!!

    def prec_loss(self, disp):  ##!!!!!!!!
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        fd = fdt.FD_torch(np.array([1.,1.,1.]))
        dfx = fd.dXc(disp[:, 0, ...])
        dfy = fd.dYc(disp[:, 1, ...])
        dfz = fd.dZc(disp[:, 2, ...])
        l2 = dfx ** 2 + dfy ** 2 + dfz ** 2
        reg = l2.mean()
        return reg * 0.5


    def kl_loss(self):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """


        # prepare inputs
        ndims = 3
        flow_mean = self.res_flow_mean
        log_sigma = self.res_log_sigma

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data

        # sigma terms
        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma  ##!!!!!!!!
        sigma_term = torch.mean(sigma_term)  ##!!!!!!!!

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda_mean * self.prec_loss(flow_mean)  # this is the jacobi loss
        if self.print_count%10==0:
            print("the loss of neg log_sigma is {},  the sigma term is {}, the loss of the prec term is {}".format((-log_sigma).mean().item(),sigma_term,prec_term))

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def recon_loss(self):
        """ reconstruction loss """
        y_pred = self.warped
        y_true = self.target
        return 1. / (self.image_sigma ** 2) * torch.mean((y_true - y_pred)**2)  ##!!!!!!!!


    def get_loss(self):
        sim_loss = self.get_sim_loss()
        reg_loss = self.scale_reg_loss()
        return sim_loss+ reg_loss


    def get_inverse_map(self,use_01=False):
        # TODO  not test yet
        print("VoxelMorph approach doesn't support analytical computation of inverse map")
        print("Instead, we compute it's numerical approximation")
        _, inverse_map,_ = self.forward(self.target, self.source)
        return inverse_map





    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()




#
# def test():
#     from mermaid.module_parameters import ParameterDict
#     cuda = torch.device('cuda:0')
#     test_cvpr = True
#     opt = ParameterDict()
#     if test_cvpr:
#         # unet = UNet_light2(2,3).to(cuda)
#         net = VoxelMorphCVPR2018([80, 192, 192],opt).to(cuda)
#     else:
#         net = VoxelMorphMICCAI2019([80,192,192],opt).to(cuda)
#     print(net)
#     with torch.enable_grad():
#         input1 = torch.randn(1, 1, 80, 192, 192).to(cuda)
#         input2 = torch.randn(1, 1, 80, 192, 192).to(cuda)
#         disp_field, warped_input1, deform_field = net(input1, input2)
#         loss = net.get_loss()
#         print("The loss is {}".format(loss))
#
#     pass
#
# if __name__ == '__main__':
#     test()
