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
    """
    implementation of loss function
    current support list:
    "l1": Lasso
    "mse": mean square error
    'ncc': normalize cross correlation
    'lncc': localized normalized lncc (here, we implement the multi-kernel localized normalized lncc)
    """
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
        elif cont_loss_type =='empty':
            self.criterion = None
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)



    def get_loss(self,output, gt, inst_weights=None, train=False):
        if self.criterion is not None:
            return self.criterion(output,gt)





class NCCLoss(nn.Module):
    """
    A implementation of the normalized cross correlation (NCC)
    """
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
    """This is an generalized LNCC; we implement multi-scale (means resolution)
    multi kernel (means size of neighborhood) LNCC.

    :param: resol_bound : type list,  resol_bound[0]> resol_bound[1] >... resol_bound[end]
    :param: kernel_size_ratio: type list,  the ratio of the current input size
    :param: kernel_weight_ratio: type list,  the weight ratio of each kernel size, should sum to 1
    :param: stride: type_list, the stride between each pixel that would compute its lncc
    :param: dilation: type_list

    Settings in json::

        "similarity_measure": {
                "develop_mod_on": false,
                "sigma": 0.5,
                "type": "lncc",
                "lncc":{
                    "resol_bound":[-1],
                    "kernel_size_ratio":[[0.25]],
                    "kernel_weight_ratio":[[1.0]],
                    "stride":[0.25,0.25,0.25],
                    "dilation":[1]
                }

    For multi-scale multi kernel, e.g.,::

        "resol_bound":[64,32],
        "kernel_size_ratio":[[0.0625,0.125, 0.25], [0.25,0.5], [0.5]],
        "kernel_weight_ratio":[[0.1,0.3,0.6],[0.3,0.7],[1.0]],
        "stride":[0.25,0.25,0.25],
        "dilation":[1,2,2] #[2,1,1]

    or for single-scale single kernel, e.g.,::

        "resol_bound":[-1],
        "kernel_size_ratio":[[0.25]],
        "kernel_weight_ratio":[[1.0]],
        "stride":[0.25],
        "dilation":[1]


    Multi-scale is controlled by "resol_bound", e.g resol_bound = [128, 64], it means if input size>128, then it would compute multi-kernel
    lncc designed for large image size,  if 64<input_size<128, then it would compute multi-kernel lncc desiged for mid-size image, otherwise,
    it would compute the multi-kernel lncc designed for small image.
    Attention! we call it multi-scale just because it is designed for multi-scale registration or segmentation problem.
    ONLY ONE scale would be activated during computing the similarity, which depends on the current input size.

    At each scale, corresponding multi-kernel lncc is implemented, here multi-kernel means lncc with different window sizes
    Loss = w1*lncc_win1 + w2*lncc_win2 ... + wn*lncc_winn, where /sum(wi) =1
    for example. when (image size) S>128, three windows sizes can be used, namely S/16, S/8, S/4.
    for easy notation, we use img_ratio to refer window size, the example here use the parameter [1./16,1./8,1.4]

    In implementation, we compute lncc by calling convolution function, so in this case, the [S/16, S/8, S/4] refers
    to the kernel size of convolution function.  Intuitively,  we would have another two parameters,
    stride and dilation. For each window size (W), we recommend using W/4 as stride. In extreme case the stride can be 1, but
    can large increase computation.   The dilation expand the reception field, set dilation as 2 would physically twice the window size.
    """

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



