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
    def __init__(self,opt,record_weight=None, imd_weight=None, score_differ=None,old_score_diff= None, manual_set = False):
        super(Loss,self).__init__()
        cont_loss_type = opt['tsk_set']['loss']['type']
        self.manual_set = manual_set
        self.record_weight = record_weight  # numpy
        self.cur_weight = None   # tensor
        self.score_differ = score_differ
        self.old_score_diff = old_score_diff
        class_num = opt['tsk_set']['extra_info']['num_label']
        if cont_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif cont_loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif cont_loss_type =='ce':
            class_weight = self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight   # update the record weight
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            ce_opt = opt['tsk_set']['loss']['ce']
            ce_opt['class_weight'] = self.cur_weight
            ce_opt['class_num'] = class_num
            self.criterion = CrossEntropyLoss(ce_opt)
        elif cont_loss_type == 'focal_loss':
            focal_loss = FocalLoss()
            class_weight =self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight   # the returneed class weighted has been normalized into sum 1
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            weight = self.cur_weight if opt['tsk_set']['loss']['focal_loss_weight_on'] else None
            focal_loss.initialize(class_num, alpha=weight, gamma=2, size_average=True)
            self.criterion = focal_loss
        elif cont_loss_type == 'dice_loss':
            dice_loss =  DiceLoss()
            class_weight = self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight  # the returneed class weighted has been normalized into sum 1
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            weight = self.cur_weight if opt['tsk_set']['loss']['dice_loss_weight_on'] else None
            dice_loss.initialize(class_num,weight)
            self.criterion =dice_loss
        elif cont_loss_type == 'gdice_loss':
            dice_loss =  GeneralizedDiceLoss()
            weight = self.get_inverse_label_density(opt)
            weight = torch.cuda.FloatTensor(weight)
            dice_loss.initialize(class_num,weight)
            self.criterion =dice_loss
        elif cont_loss_type == 'tdice_loss':
            dice_loss =  TverskyLoss()
            class_weight = self.cal_class_weight(opt, self.record_weight)
            self.record_weight = class_weight  # the returneed class weighted has been normalized into sum 1
            self.cur_weight = torch.cuda.FloatTensor(class_weight)
            weight = self.cur_weight if opt['tsk_set']['loss']['tdice_loss_weight_on'] else None
            dice_loss.initialize(class_num,weight)
            self.criterion =dice_loss
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
        if not manual_set:
            self.record_weight = None


    def get_loss(self,output, gt, inst_weights=None, train=False):
        return self.criterion(output, gt,inst_weights=inst_weights, train=train)


    def get_inverse_label_density(self,opt):
        label_density = opt['tsk_set']['extra_info'][('label_density', [], 'label_density')]
        label_density = np.asarray(label_density)
        class_weight = 1.0 / label_density / np.sum(1.0 / label_density)
        class_weight[0] = np.average(class_weight[1:])
        return class_weight


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
            rand_resid_update = opt['tsk_set']['loss'][('rand_resid_update', False, 'using residue weight')]
            only_bg_avg_update = opt['tsk_set']['loss'][('only_bg_avg_update', False, 'using only_bg_avg_update weight')]
            only_bg_avg_log_update = opt['tsk_set']['loss'][('only_bg_avg_log_update', False, 'using only_bg_avg_log_update weight')]
            factor_update = opt['tsk_set']['loss'][('factor_update', False, 'using factor weight ')]
            # if continuous_update:
            #     residue_weight = opt['tsk_set']['loss'][('residue_weight', [0] * class_num, 'residue weight')]  # from 34
            #     residue_weight = np.asarray(residue_weight)
            #     m = opt['tsk_set']['loss'][('residue_weight_momentum', 0.3, 'residue weight')]
            #     class_weight = record_weight if record_weight is not None else class_weight
            #     re_class_weight = (1-m)*class_weight + residue_weight * m
            if log_update:
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [0] * class_num, 'residue weight')]  # from 34
                residue_weight = np.log1p((np.asarray(residue_weight))+1.5)
                re_class_weight = residue_weight
                ga = opt['tsk_set']['loss'][('residue_weight_gama', 1., 'residue weight gama')]
                ap = opt['tsk_set']['loss'][('residue_weight_alpha', 1., 'residue weight alpha')]
                #re_class_weight = ap*np.log1p(class_weight) + np.log1p(residue_weight+0.5)*ga # from task110
            if only_resid_update:  # from task51 add sm
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                class_weight = np.asarray([1.] * class_num)
                ga = opt['tsk_set']['loss'][('residue_weight_gama', 1., 'residue weight gama')]
                # residue_weight =np.log1p(class_weight)+ ga* np.log1p((np.asarray(residue_weight)))
                residue_weight = np.log1p((np.asarray(residue_weight))+0.15)   # fromm task52  from task68_2 # remove ga from task110
                re_class_weight = residue_weight

            if rand_resid_update:  # from task51 add sm
                residue_weight = opt['tsk_set']['loss'][('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                rand_ratio = opt['tsk_set']['loss'][('rand_ratio',0.5, 'rand_ratio')]  # from 34
                print("current rand_raito is {}".format(rand_ratio))
                mask_num = int((1.-rand_ratio)* len(residue_weight))
                masked_label = residue_weight
                if self.manual_set: # here mask_num refer to the label need to be masked, background 0 should be left
                    masked_label = np.argsort(residue_weight)[1:mask_num]   # fromm task52  from task68_2 # remove ga from task110
                return masked_label
            if only_bg_avg_update:
                residue_weight = opt['tsk_set']['loss'][
                    ('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                residue_weight =np.asarray(residue_weight)  # fromm task52  from task68_2 # remove ga from task110
                residue_weight[0] = np.average(residue_weight[1:])
                re_class_weight = residue_weight
            if only_bg_avg_log_update:
                residue_weight = opt['tsk_set']['loss'][
                    ('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                residue_weight =np.asarray(residue_weight)  # fromm task52  from task68_2 # remove ga from task110
                residue_weight[0] = np.average(residue_weight[1:])
                re_class_weight = np.log1p(residue_weight+0.1)  # 4.9
            if factor_update:
                residue_weight = opt['tsk_set']['loss'][
                    ('residue_weight', [1] * class_num, 'residue weight')]  # from 34
                residue_weight = np.asarray(residue_weight)  # fromm task52
                factor = 1
                print("using factor update, the current factor is {}".format(factor))
                record_weight = record_weight if record_weight is not None else np.log1p(residue_weight+0.05)  # the residue only be used as the intialization
                score_differ = self.score_differ if self.score_differ is not None else class_weight  # the residue only be used as the intialization
                old_score_diff = self.old_score_diff

                if old_score_diff is not None:
                    direction = np.sign(score_differ) * np.sign(old_score_diff)
                else:
                    direction = np.sign(score_differ)
                scale = factor * direction *np.abs(score_differ)
                print("the update direction is {}".format( direction))
                re_class_weight = np.exp(scale)* record_weight
                print( "current exp scale:{}".format(np.exp(scale)))
                print( "the current factor based weight is {}".format(re_class_weight))
                return re_class_weight



            if continuous_update:
                m = opt['tsk_set']['loss'][('residue_weight_momentum', 0.3, 'residue weight')]
                record_weight = record_weight if record_weight is not None else class_weight
                print("the record weight is:{}, with momentum:{}".format(record_weight,m))
                re_class_weight =re_class_weight / np.sum(re_class_weight)
                re_class_weight =m*record_weight + re_class_weight * (1-m)

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

    def forward(self, inputs, targets, weight= None, inst_weights=None, train=None):
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



class DiceLoss(nn.Module):
    def initialize(self, class_num, weight = None):
        self.class_num = class_num
        self.class_num = class_num
        if weight is None:
            self.weight = Variable(torch.ones(class_num, 1))/self.class_num
        else:
            if isinstance(weight, Variable):
                self.weight = weight
            else:
                self.weight = Variable(weight)
        self.weight = torch.squeeze(self.weight)
    def forward(self,input, target, inst_weights=None,train=None):
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

        den1 = probs#*probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = target#*target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))


        dice = 2 * (num / (den1 + den2))
        dice = self.weight.expand_as(dice) * dice
        dice_eso = dice
        # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
        return dice_total



class GeneralizedDiceLoss(nn.Module):
    def initialize(self, class_num, weight=None):
        self.class_num = class_num
        if weight is None:
            self.weight = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(weight, Variable):
                self.weight = weight
            else:
                self.weight = Variable(weight)
        self.weight = torch.squeeze(self.weight)
    def forward(self,input, target,inst_weights=None,train=None):
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
        num = torch.sum(num, dim=2)  # batch x ch

        den1 = probs
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)  # batch x ch

        den2 = target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)  # batch x ch
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))
        weights = self.weight.expand_as(den1)

        dice = 2 * (torch.sum(weights*num,dim=1) / torch.sum(weights*(den1 + den2),dim=1))
        dice_eso = dice
        # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total




class TverskyLoss(nn.Module):
    def initialize(self, class_num, weight=None, alpha=0.5, beta=0.5):
        self.class_num = class_num
        if weight is None:
            self.weight = Variable(torch.ones(class_num, 1))/self.class_num
        else:
            if isinstance(weight, Variable):
                self.weight = weight
            else:
                self.weight = Variable(weight)
        self.weight = torch.squeeze(self.weight)
        self.alpha = alpha
        self.beta = beta
        print("the weight of Tversky loss is  {}".format(weight))

    def forward(self,input, target,inst_weights=None, train=None):
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


        den1 = probs*(1-target)
        den1 = den1.view(den1.shape[0], den1.shape[1], -1)
        den1 = torch.sum(den1, dim=2)

        den2 = (1-probs)*target
        den2 = den1.view(den2.shape[0], den2.shape[1], -1)
        den2 = torch.sum(den2, dim=2)
        # print("den1:{}".format(sum(sum(den1))))
        # print("den2:{}".format(sum(sum(den2/den1))))


        dice = 2 * (num / (num + self.alpha*den1 + self.beta*den2))
        dice = self.weight.expand_as(dice) * dice

        dice_eso = dice
        #dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total








class CrossEntropyLoss(nn.Module):
    def __init__(self, opt, imd_weight=None):
        # To Do,  add dynamic weight
        super(CrossEntropyLoss,self).__init__()
        no_bg = opt[('no_bg',False,'exclude background')]
        weighted = opt[('weighted',True,'  weighted the class')]
        reduced = opt[('reduced',True,'  reduced the class')]
        self.mask = None #opt[('mask',None, 'masked other label')]
        class_num = opt['class_num']

        if no_bg:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        if weighted:
            class_weight = opt['class_weight']if imd_weight is None else imd_weight
            if class_weight is not None and not (len(class_weight)< class_num):
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduce = reduced)
                self.mask=None
            else:  # this is the case for using random mask, the class weight here refers to the label need be masked
                self.mask = class_weight
                print("the current mask is {}".format(self.mask))
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduce = reduced)
        self.n_class = class_num
    def forward(self, input, gt, inst_weights= None, train=False):
        """
        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        if self.mask is not None and train:
            for m in self.mask:
                gt[gt==m]=0
        if len(input.shape)==5:
            output_flat = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_class)
        else:
            output_flat = input
        truths_flat = gt.view(-1)
        if inst_weights is None:
            return self.loss_fn(output_flat,truths_flat)
        else:
            return torch.mean( inst_weights.view(-1)*self.loss_fn(output_flat,truths_flat))




