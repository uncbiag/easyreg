# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Functions
###############################################################################

class Loss(object):
    def __init__(self,opt):
        super(Loss,self).__init__()
        class_num = opt['tsk_set']['extra_info']['num_label']
        cont_loss_type = opt['tsk_set']['loss']
        if cont_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif cont_loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif cont_loss_type =='ce':
            self.criterion = CrossEntropyLoss(class_num)
        elif cont_loss_type == 'focal_loss':
            focal_loss = FocalLoss()
            focal_loss.initialize(class_num, alpha=None, gamma=2, size_average=True)
            self.criterion = focal_loss
        elif cont_loss_type == 'dice_loss':
            self.criterion = DiceLoss()
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)

    def get_loss(self,output, gt):
        return self.criterion(output, gt)



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

    def initialize(self, class_num, alpha=None, gamma=2, size_average=True):
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
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
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

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
        targ_one_hot = torch.zeros(in_sz[0],in_sz[1],extra_dim)
        targ_one_hot.scatter_(1,target.view(in_sz[0],1,extra_dim),1)
        target = targ_one_hot.view(in_sz).contiguous()
        assert input.size() == target.size(), "Input sizes must be equal."
        uniques = np.unique(target.numpy())
        assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

        probs = F.softmax(input)
        num = probs * target
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)

        den1 = probs * probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = target * target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)

        dice = 2 * (num / (den1 + den2))
        dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total



class CrossEntropyLoss(nn.Module):
    def __init__(self, class_num):
        super(CrossEntropyLoss,self).__init__()
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




