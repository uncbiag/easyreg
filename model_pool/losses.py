# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import mermaid.finite_differences as fdt

###############################################################################
# Functions
###############################################################################

class Loss(object):
    def __init__(self,opt):
        super(Loss,self).__init__()
        cont_loss_type = opt['tsk_set']['loss']['type']
        if cont_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif cont_loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif cont_loss_type =='ncc':
            self.criterion = NCCLoss()
        elif cont_loss_type =='lncc':
            lncc =  LNCCLoss()
            lncc.initialize()
            self.criterion =lncc
        elif cont_loss_type =='morph_cvpr':
            vm =  MorphCVPR()
            vm.initialize()
            self.criterion =vm
        elif cont_loss_type =='empty':
            self.criterion = None
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)



    def get_loss(self,output, gt, inst_weights=None, train=False):
        if self.criterion is not None:
            return self.criterion(output,gt)





class NCCLoss(nn.Module):
    def forward(self,input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1).view(input.shape[0],1)
        target_minus_mean = target - torch.mean(target, 1).view(input.shape[0],1)
        nccSqr = ((input_minus_mean * target_minus_mean).mean(1)) / torch.sqrt(
                    ((input_minus_mean ** 2).mean(1)) * ((target_minus_mean ** 2).mean(1)))
        nccSqr =  nccSqr.mean()

        return (1 - nccSqr)*input.shape[0]

class LNCCLoss(nn.Module):
    def initialize(self, kernel_sz = [9,9,9], voxel_weights = None):
        pass


    def __stepup(self,img_sz, use_multi_scale=True):
        max_scale  = min(img_sz)
        if use_multi_scale:
            if max_scale>128:
                self.scale = [int(max_scale/16), int(max_scale/8), int(max_scale/4)]
                self.scale_weight = [0.1, 0.3, 0.6]
                self.dilation = [2,2,2]


            elif max_scale>64:
                self.scale = [int(max_scale / 4), int(max_scale / 2)]
                self.scale_weight = [0.3,0.7]
                self.dilation = [2,2]
            else :
                self.scale = [int(max_scale / 2)]
                self.scale_weight = [1.0]
                self.dilation = [1]
        else:
            self.scale_weight =  [int(max_scale/4)]
            self.scale_weight = [1.0]
        self.num_scale = len(self.scale)
        self.kernel_sz = [[scale for _ in range(3)] for scale in self.scale]
        self.step = [[max(int((ksz + 1) / 4),1) for ksz in self.kernel_sz[scale_id]] for scale_id in range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        self.conv = F.conv3d




    def forward(self, input, target):
        self.__stepup(img_sz=list(input.shape[2:]))
        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(input.shape[0], -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(input.shape[0],
                                                                          -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(input.shape[0],
                                                                           -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(
                input.shape[0], -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                input.shape[0], -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc_total += lncc * self.scale_weight[scale_id]

        return lncc_total*(input.shape[0])






class MorphCVPR(nn.Module):
    """
    loss used in voxel morph cvpr 2018
    """

    def initialize(self, kernel_sz=[9, 9, 9], voxel_weights=None):
        pass

    def __stepup(self, img_sz, use_multi_scale=False):
        max_scale = min(img_sz)
        self.scale=[9]
        self.scale_weight = [1.0]
        self.dilation = [1]
        self.num_scale = len(self.scale)
        self.kernel_sz = [[scale for _ in range(3)] for scale in self.scale]
        self.step = [[max(int((ksz + 1) / 4), 1) for ksz in self.kernel_sz[scale_id]] for scale_id in
                     range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        self.conv = F.conv3d

    def forward(self, input, target):
        self.__stepup(img_sz=list(input.shape[2:]))
        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(input.shape[0], -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(input.shape[0],
                                                                          -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(input.shape[0],
                                                                           -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(
                input.shape[0], -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                input.shape[0], -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc_total += lncc * self.scale_weight[scale_id]

        return lncc_total * (input.shape[0])



class MorphMICCAI(nn.Module):
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, img_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = img_shape
        self.spacing = 1. / (np.array(img_shape) - 1) ##!!!!!!!!

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
        fd = fdt.FD_torch(self.spacing * 2)
        dfx = fd.dXc(disp[:, 0, ...])
        dfy = fd.dYc(disp[:, 1, ...])
        dfz = fd.dZc(disp[:, 2, ...])
        l2 = dfx ** 2 + dfy ** 2 + dfz ** 2
        reg = l2.mean()
        return reg * 0.5


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.shape) - 2 ##!!!!!!!!
        mean = y_pred[:,0:ndims,..., ]
        log_sigma = y_pred[:,ndims:,...]  ##!!!!!!!!

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)  # 1, 96, 40,40 3

        # sigma terms
        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma  ##!!!!!!!!
        sigma_term = torch.mean(sigma_term)  ##!!!!!!!!

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)  # this is the jacobi loss

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma ** 2) * torch.mean((y_true - y_pred)**2)  ##!!!!!!!!
