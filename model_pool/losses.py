# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model_pool.metrics import get_multi_metric

###############################################################################
# Functions
###############################################################################

class Loss(object):
    def __init__(self,opt,record_weight=None, imd_weight=None):
        super(Loss,self).__init__()
        cont_loss_type = opt['tsk_set']['loss']['type']
        self.record_weight = record_weight  # numpy
        self.cur_weight = None   # tensor
        class_num = opt['tsk_set']['extra_info']['num_label']
        if cont_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif cont_loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif cont_loss_type =='ce':
            class_weight = self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            ce_opt = opt['tsk_set']['loss']['ce']
            ce_opt['class_weight'] = self.cur_weight
            ce_opt['class_num'] = class_num
            self.criterion = CrossEntropyLoss(ce_opt)
        elif cont_loss_type == 'focal_loss':
            focal_loss = FocalLoss()
            class_weight =self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            alpha = self.cur_weight if opt['tsk_set']['loss']['focal_loss_weight_on'] else None
            focal_loss.initialize(class_num, alpha=alpha, gamma=2, size_average=True)
            self.criterion = focal_loss
        elif cont_loss_type == 'dice_loss':
            self.criterion = DiceLoss()
        elif cont_loss_type == 'ce_imd':
            #assert imd_weight is not None," in ce_imd mode, imd weight should never be none"
            ce_opt = opt['tsk_set']['loss']['ce']
            if imd_weight is None:
                print('Warning, current imd_weight is None')
            ce_opt['class_num'] = class_num
            ce_opt['class_weight'] = None
            self.criterion = CrossEntropyLoss(ce_opt,imd_weight)
        elif cont_loss_type == 'focal_loss_imd':
            #assert imd_weight is not None," in focal_loss_imd mode, imd weight should never be none"
            focal_loss = FocalLoss()
            alpha = imd_weight
            focal_loss.initialize(class_num, alpha=alpha, gamma=2, size_average=True, verbose=False)
            self.criterion = focal_loss
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)


    def get_loss(self,output, gt):
        return self.criterion(output, gt)




    def cal_class_weight(self,opt,record_weight):
        class_num = opt['tsk_set']['extra_info']['num_label']
        class_weight=None
        density_weight_on = opt['tsk_set']['loss'][('density_weight_on', False, 'using density weight')]
        if density_weight_on:
            label_density = opt['tsk_set']['extra_info'][('label_density', [], 'label_density')]
            label_density = np.asarray(label_density)
            class_weight = 1.0 / label_density / np.sum(1.0 / label_density)
        else:
            class_weight = np.asarray([1./float(class_num)]*class_num) #from 34


        residue_weight_on = opt['tsk_set']['loss'][('residue_weight_on', False, 'using residue weight')]
        if residue_weight_on:

            continuous_update = opt['tsk_set']['loss'][('continuous_update', False, 'using residue weight')]
            log_update = opt['tsk_set']['loss'][('log_update', False, 'using residue weight')]
            only_resid_update = opt['tsk_set']['loss'][('only_resid_update', False, 'using residue weight')]
            if continuous_update:
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [0] * class_num, 'residue weight')]  # from 34
                residue_weight = np.asarray(residue_weight)
                m = opt['tsk_set']['loss'][('residue_weight_momentum', 0.1, 'residue weight')]
                class_weight = record_weight if record_weight is not None else class_weight
                re_class_weight = (1-m)*class_weight + residue_weight * m
            if log_update:
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [0] * class_num, 'residue weight')]  # from 34
                residue_weight = np.asarray(residue_weight)
                ga = opt['tsk_set']['loss'][('residue_weight_gama', 1., 'residue weight gama')]
                ap = opt['tsk_set']['loss'][('residue_weight_alpha', 1., 'residue weight alpha')]
                re_class_weight = ap*np.log1p(class_weight) + np.log1p(residue_weight+0.5)*ga # from task110
            if only_resid_update:  # from task51 add sm
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                class_weight = np.asarray([1.] * class_num)
                ga = opt['tsk_set']['loss'][('residue_weight_gama', 1., 'residue weight gama')]
                # residue_weight =np.log1p(class_weight)+ ga* np.log1p((np.asarray(residue_weight)))
                residue_weight = np.log1p((np.asarray(residue_weight))+0.5)   # fromm task52  from task68_2 # remove ga from task110
                re_class_weight = residue_weight
            print("class_weight:{}".format(class_weight))
            print("residue_weight:{}".format(residue_weight))
            class_weight = re_class_weight / np.sum(re_class_weight) #
            print("normalized_weight:{}".format(class_weight))
        return class_weight



class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def initialize(self, class_num, alpha=None, gamma=2, size_average=True, verbose=True):
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.alpha = torch.squeeze(self.alpha)
        if verbose:
            print("the alpha of focal loss is  {}".format(alpha))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        """

        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1)

        P = F.softmax(inputs,dim=1)
        ids = targets.view(-1)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]

        log_p = - F.cross_entropy(inputs, targets,reduce=False)
        probs = F.nll_loss(P, targets,reduce=False)
        # print(probs)
        # print(log_p)
        # print(torch.pow((1 - probs), self.gamma))
        # print(alpha)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    def debugging(self):
        alpha = torch.rand(21, 1)
        print(alpha)
        FL = FocalLoss(class_num=5, gamma=0)
        CE = nn.CrossEntropyLoss()
        N = 4
        C = 5
        inputs = torch.rand(N, C)
        targets = torch.LongTensor(N).random_(C)
        inputs_fl = Variable(inputs.clone(), requires_grad=True)
        targets_fl = Variable(targets.clone())

        inputs_ce = Variable(inputs.clone(), requires_grad=True)
        targets_ce = Variable(targets.clone())
        print('----inputs----')
        print(inputs)
        print('---target-----')
        print(targets)

        fl_loss = FL(inputs_fl, targets_fl)
        ce_loss = CE(inputs_ce, targets_ce)
        print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
        fl_loss.backward()
        ce_loss.backward()
        # print(inputs_fl.grad.data)
        print(inputs_ce.grad.data)



class DiceLoss(nn.Module):
    def forward(self,input, target):
        """
        input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
        target is a Bx....   range 0,1....N_label
        """
        in_sz = input.size()
        from functools import reduce
        extra_dim = reduce(lambda x,y:x*y,in_sz[2:])
        targ_one_hot = Variable(torch.zeros(in_sz[0],in_sz[1],extra_dim)).cuda()
        targ_one_hot.scatter_(1,target.view(in_sz[0],1,extra_dim),1.)
        target = targ_one_hot.view(in_sz).contiguous()
        probs = F.softmax(input,dim=1)
        num = probs*target
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)

        den1 = probs*probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = target*target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)

        dice = 2 * (num / (den1 + den2))
        dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total



class CrossEntropyLoss(nn.Module):
    def __init__(self, opt, imd_weight=None):
        # To Do,  add dynamic weight
        super(CrossEntropyLoss,self).__init__()
        no_bg = opt[('no_bg',False,'exclude background')]
        weighted = opt[('weighted',True,'  weighted the class')]
        class_num = opt['class_num']

        if no_bg:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        if weighted:
            class_weight = opt['class_weight']if imd_weight is None else imd_weight
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.n_class = class_num
    def forward(self, input, gt):
        """
        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        output_flat = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_class)
        truths_flat = gt.view(-1)
        return self.loss_fn(output_flat,truths_flat)




