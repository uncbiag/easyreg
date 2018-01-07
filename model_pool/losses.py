import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Functions
###############################################################################

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self,output, gt):
        return self.criterion(output, gt)





class FocalLoss(nn.Module):
    r"""
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
        N = inputs.size(0)
        print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
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
        target is a 1-hot representation of the groundtruth, shoud have same size as the input, which means the value in target should be 0,1....N_label
        """
        in_sz = input.size()
        from functools import reduce
        extra_dim = reduce(lambda x,y:x*y,in_sz[2:])
        targ_one_hot = torch.zeros(in_sz[0],in_sz[1],extra_dim)
        targ_one_hot.scatter_(1,target.view(in_sz[0],in_sz[1],extra_dim),1)
        targ_one_hot = targ_one_hot.view(in_sz).contiguous()
        assert input.size() == target.size(), "Input sizes must be equal."
        uniques = np.unique(target.numpy())
        assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

        probs = F.softmax(input)
        num = probs * target  # b,c,h,w--p*g
        num = num.view(num.shape[0],num.shape[1],-1)
        num = torch.sum(num, dim=2)  # b,c,h

        den1 = probs * probs  # --p^2
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)  # b,c,h

        den2 = target * target  # --g^2
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)  # b,c,hb,c

        dice = 2 * (num / (den1 + den2))
        dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total


class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        #################################################################33
        #  should be replaced by 3D medical network
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss



def init_loss(opt):
    disc_loss = None
    cont_loss_type = opt.cont_loss_type
    content_loss = ContentLoss()
    if cont_loss_type == 'l1':
        content_loss.initialize(nn.L1Loss())
    elif cont_loss_type == 'l2':
        content_loss.initialize(nn.MSELoss())
    elif cont_loss_type =='ce':
        content_loss.initialize(nn.CrossEntropyLoss())
    elif cont_loss_type == 'focal_loss':
        class_num = opt.class_num
        focal_loss = FocalLoss().initialize(class_num, alpha=None, gamma=2, size_average=True)
        content_loss = focal_loss
    elif cont_loss_type == 'dice_loss':
        content_loss = DiceLoss()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    return disc_loss, content_loss
