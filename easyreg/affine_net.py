from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .modules import *
from .net_utils import Bilinear
from torch.utils.checkpoint import checkpoint
from .utils import sigmoid_decay
from .losses import NCCLoss



class AffineNetSym(nn.Module):
    """
    A multi-step symmetirc -force affine network

    at each step. the network would update the affine parameter
    the advantage is we don't need to warp the image for several time (only interpolated by the latest affine transform) as the interpolation would diffuse the image

    """
    def __init__(self, img_sz=None, opt=None):
        super(AffineNetSym, self).__init__()
        self.img_sz = img_sz
        """ the image sz  in numpy coord"""
        self.dim = len(img_sz)
        """ the dim of image"""
        self.step = opt['tsk_set']['reg']['affine_net'][('affine_net_iter',1,'num of step')]
        """ the num of step"""
        self.step_record = self.step
        """ a copy on step"""
        self.using_complex_net = opt['tsk_set']['reg']['affine_net'][('using_complex_net',True,'use complex version of affine net')]
        """if true, use complex version of affine net"""
        self.acc_multi_step_loss = opt['tsk_set']['reg']['affine_net'][('acc_multi_step_loss',False,'accumulate loss at each step')]
        """accumulate loss from each step"""
        self.initial_reg_factor = opt['tsk_set']['reg']['affine_net'][('initial_reg_factor', 10, 'initial regularization factor')]
        """initial regularization factor"""
        self.min_reg_factor = opt['tsk_set']['reg']['affine_net'][('min_reg_factor', 1e-3, 'minimum regularization factor')]
        """minimum regularization factor"""
        self.epoch_activate_multi_step = opt['tsk_set']['reg']['affine_net'][('epoch_activate_multi_step',-1,'epoch to activate multi-step affine')]
        """epoch to activate multi-step affine"""
        self.reset_lr_for_multi_step = opt['tsk_set']['reg']['affine_net'][('reset_lr_for_multi_step',False,'if True, reset learning rate when multi-step begins')]
        """if True, reset learning rate when multi-step begins"""
        self.lr_for_multi_step = opt['tsk_set']['reg']['affine_net'][('lr_for_multi_step',5e-5,'if reset_lr_for_multi_step, reset learning rate into # when multi-step begins')]
        """if reset_lr_for_multi_step, reset learning rate into # when multi-step begins"""
        self.epoch_activate_sym = opt['tsk_set']['reg']['affine_net'][('epoch_activate_sym',-1,'epoch to activate symmetric forward')]
        """epoch to activate symmetric forward"""
        self.sym_factor = opt['tsk_set']['reg']['affine_net'][('sym_factor', 1., 'the factor of symmetric loss')]
        """ the factor of symmetric loss"""
        self.mask_input_when_compute_loss = opt['tsk_set']['reg']['affine_net'][('mask_input_when_compute_loss', False, 'mask_input_when_compute_loss')]
        """ mask input when compute loss"""
        self.epoch_activate_sym_loss = opt['tsk_set']['reg']['affine_net'][('epoch_activate_sym_loss',-1,'the epoch to take symmetric loss into backward , only if epoch_activate_sym and epoch_activate_sym_loss')]
        """ the epoch to take symmetric loss into backward , only if epoch_activate_sym and epoch_activate_sym_loss"""
        self.epoch_activate_extern_loss = opt['tsk_set']['reg']['affine_net'][('epoch_activate_extern_loss',-1,'epoch to activate the external loss which will replace the default ncc loss')]
        """epoch to activate the external loss which will replace the default ncc loss"""

        self.affine_fc_size = opt['tsk_set']['reg']['affine_net'][(
            'affine_fc_size', 720, 'size of the full connected layer, changes depending on input size')]

        """epoch to activate the external loss which will replace the default ncc loss"""
        self.affine_gen = Affine_unet_im(fc_size=self.affine_fc_size) if self.using_complex_net else Affine_unet()

        """ the affine network output the affine parameter"""
        self.affine_param = None
        """ the affine parameter with the shape of Nx 12 for 3d transformation"""
        self.affine_cons= AffineConstrain()
        """ the func return regularization loss on affine parameter"""
        self.id_map= gen_identity_map(self.img_sz).cuda()
        """ the identity map"""
        self.gen_identity_ap()
        """ generate identity affine parameter"""

        ################### init variable  #####################3
        self.iter_count = 0
        """the num of iteration"""
        self.using_multi_step = True
        """ set multi-step on"""
        self.zero_boundary = True
        """ zero boundary is used for interpolated images"""
        self.epoch = -1
        """ the current epoch"""
        self.ncc = NCCLoss()
        """ normalized cross correlation loss"""
        self.extern_loss = None
        """ external loss used during training"""
        self.compute_loss = True
        """ compute loss, set true during affine network's training"""




    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.extern_loss = loss_fn


    def set_cur_epoch(self, cur_epoch):
        """ set current epoch"""
        self.epoch = cur_epoch


    def set_step(self,step):
        """set num of step"""
        self.step = step
        print("the step in affine network is set to {}".format(step))

    def check_if_update_lr(self):
        """
        check if the learning rate need to be updated
        during affine training, both epoch_activate_multi_step and reset_lr_for_multi_step are activated
        the learning rate would be set to reset_lr_for_multi_step

        """
        if self.epoch == self.epoch_activate_multi_step and self.reset_lr_for_multi_step:
            lr = self.lr_for_multi_step
            self.reset_lr_for_multi_step = False
            print("the lr is change into {} due to the activation of the multi-step".format(lr))
            return True, lr
        else:
            return False, None

    def gen_affine_map(self,Ab):
        """
        generate the affine transformation map with regard to affine parameter

        :param Ab: affine parameter
        :return: affine transformation map
        """
        Ab = Ab.view( Ab.shape[0],4,3) # 3d: (batch,4,3)
        id_map = self.id_map.view(self.dim, -1)
        affine_map = None
        if self.dim == 3:
            affine_map = torch.matmul( Ab[:,:3,:], id_map)
            affine_map = Ab[:,3,:].contiguous().view(-1,3,1) + affine_map
            affine_map= affine_map.view([Ab.shape[0]] + list(self.id_map.shape))
        return affine_map


    def update_affine_param(self, cur_af, last_af): # A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2
        """
        update the current affine parameter A2 based on last affine parameter A1
         A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2, results in the composed affine parameter A3=(A2A1, A2*b1+b2)

        :param cur_af: current affine parameter
        :param last_af: last affine parameter
        :return: composed affine parameter A3
        """
        cur_af = cur_af.view(cur_af.shape[0], 4, 3)
        last_af = last_af.view(last_af.shape[0],4,3)
        updated_af = torch.zeros_like(cur_af.data).cuda()
        if self.dim==3:
            updated_af[:,:3,:] = torch.matmul(cur_af[:,:3,:],last_af[:,:3,:])
            updated_af[:,3,:] = cur_af[:,3,:] + torch.squeeze(torch.matmul(cur_af[:,:3,:], torch.transpose(last_af[:,3:,:],1,2)),2)
        updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
        return updated_af

    def get_inverse_affine_param(self, affine_param):
        """
        A2(A1*x+b1) +b2= A2A1*x + A2*b1+b2 = x    A2= A1^-1, b2 = - A2^b1

        """
        affine_param = affine_param.view(affine_param.shape[0], 4, 3)
        inverse_param = torch.zeros_like(affine_param.data).cuda()
        for n in range(affine_param.shape[0]):
            tm_inv = torch.inverse(affine_param[n, :3, :])
            inverse_param[n, :3, :] = tm_inv
            inverse_param[n, 3, :] = - torch.matmul(tm_inv, affine_param[n, 3, :])
        inverse_param = inverse_param.contiguous().view(affine_param.shape[0], -1)
        return inverse_param

    def __get_inverse_map(self):
        sym_on = self.epoch>= self.epoch_activate_sym
        affine_param  = self.affine_param[:self.n_batch] if sym_on else self.affine_param
        inverse_affine_param = self.get_inverse_affine_param(affine_param)
        inverse_map = self.gen_affine_map(inverse_affine_param)
        return inverse_map

    def get_inverse_map(self,use_01=False):
        """
        get the inverse map

        :param use_01: if ture, get the map in [0,1] coord else in [-1,1] coord
        :return: the inverse map
        """
        inverse_map = self.__get_inverse_map()
        if inverse_map is not None:
            if use_01:
                return (inverse_map+1)/2
            else:
                return inverse_map
        else:
            return None

    def gen_identity_ap(self):
        """
        get the idenityt affine parameter

        :return:
        """
        self.affine_identity = torch.zeros(12).cuda()
        self.affine_identity[0] = 1.
        self.affine_identity[4] = 1.
        self.affine_identity[8] = 1.

    def compute_symmetric_reg_loss(self,affine_param, bias_factor=1.):
        """
        compute the symmetry loss
        s-t transform (a,b), t-s transform (c,d), then assume transform from t-s-t
        a(cy+d)+b = acy +ad+b =y
        then ac = I, ad+b = 0
        the l2 loss is taken to constrain the above two terms
        ||ac-I||_2^2 +  bias_factor *||ad+b||_2^2

        :param bias_factor: the factor on the translation term
        :return: the symmetry loss (average on batch)
        """

        ap_st, ap_ts  = affine_param

        ap_st = ap_st.view(-1, 4, 3)
        ap_ts = ap_ts.view(-1, 4, 3)
        ac = None
        ad_b = None
        #########  check if ad_b is right  #####
        if self.dim == 3:
            ac = torch.matmul(ap_st[:, :3, :], ap_ts[:, :3, :])
            ad_b = ap_st[:, 3, :] + torch.squeeze(
                torch.matmul(ap_st[:, :3, :], torch.transpose(ap_ts[:, 3:, :], 1, 2)), 2)
        identity_matrix = self.affine_identity.view(4,3)[:3,:3]

        linear_transfer_part = torch.sum((ac-identity_matrix)**2)
        translation_part = bias_factor * (torch.sum(ad_b**2))

        sym_reg_loss = linear_transfer_part + translation_part
        if self.iter_count %10 ==0:
            print("linear_transfer_part:{}, translation_part:{}, bias_factor:{}".format(linear_transfer_part.cpu().data.numpy(), translation_part.cpu().data.numpy(),bias_factor))
        return sym_reg_loss/ap_st.shape[0]


    def sim_loss(self,loss_fn,warped,target):
        """
        compute the similarity loss

        :param loss_fn: the loss function
        :param output: the warped image
        :param target: the target image
        :return: the similarity loss average on batch
        """
        loss_fn = self.ncc if self.epoch < self.epoch_activate_extern_loss else loss_fn
        sim_loss = loss_fn(warped,target)
        return sim_loss / warped.shape[0]


    def scale_sym_reg_loss(self,affine_param, sched='l2'):
        """
        in symmetric forward, compute regularization loss of  affine parameters,
        l2: compute the l2 loss between the affine parameter and the identity parameter
        det: compute the determinant of the affine parameter, which prefers to rigid transformation

        :param sched: 'l2' , 'det'
        :return: the regularization loss on batch
        """
        loss = self.scale_multi_step_reg_loss(affine_param[0],sched) + self.scale_multi_step_reg_loss(affine_param[1],sched)
        return loss

    def scale_multi_step_reg_loss(self,affine_param, sched='l2'):
        """
        compute regularization loss of  affine parameters,
        l2: compute the l2 loss between the affine parameter and the identity parameter
        det: compute the determinant of the affine parameter, which prefers to rigid transformation

        :param sched: 'l2' , 'det'
        :return: the regularization loss on batch
        """
        weight_mask = torch.ones(4,3).cuda()
        bias_factor = 1.0
        weight_mask[3,:]=bias_factor
        weight_mask = weight_mask.view(-1)
        if sched == 'l2':
            return torch.sum((self.affine_identity - affine_param) ** 2 *weight_mask)\
                   / (affine_param.shape[0])
        elif sched == 'det':
            mean_det = 0.
            for i in range(affine_param.shape[0]):
                affine_matrix = affine_param[i, :9].contiguous().view(3, 3)
                mean_det += torch.det(affine_matrix)
            return mean_det / affine_param.shape[0]

    def get_factor_reg_scale(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        epoch_for_reg = self.epoch if self.epoch < self.epoch_activate_multi_step else self.epoch - self.epoch_activate_multi_step
        factor_scale = self.initial_reg_factor if self.epoch < self.epoch_activate_multi_step else self.initial_reg_factor/100
        static_epoch = 10 if self.epoch < self.epoch_activate_multi_step else 1
        min_threshold = self.min_reg_factor
        decay_factor = 3
        factor_scale = float(
            max(sigmoid_decay(epoch_for_reg, static=static_epoch, k=decay_factor) * factor_scale, min_threshold))
        return factor_scale



    def compute_overall_loss(self, loss_fn, output, target,affine_map,moving_mask=None,target_mask=None):
        """
        compute the overall loss for affine tranning
        overall loss = multi-step similarity loss + symmetry loss + regularization loss

        :param loss_fn: loss function to compute the similaritysym_reg_loss
        :param output: warped image
        :param target:target image
        :return:overall loss
        """
        if self.mask_input_when_compute_loss and moving_mask is not None and target_mask is not None:
            affine_mask = Bilinear(self.zero_boundary)(moving_mask, affine_map)
            output = output*affine_mask
            target = target*target_mask
        sim_loss = self.sim_loss(loss_fn.get_loss,output, target)
        sym_on = self.epoch>= self.epoch_activate_sym
        affine_param = (self.affine_param[:self.n_batch], self.affine_param[self.n_batch:]) if sym_on else self.affine_param
        sym_reg_loss = self.compute_symmetric_reg_loss(affine_param,bias_factor=1.) if  sym_on else 0.
        scale_reg_loss = self.scale_sym_reg_loss(affine_param, sched = 'l2') if sym_on else self.scale_multi_step_reg_loss(affine_param, sched='l2')
        factor_scale = self.get_factor_reg_scale()
        factor_sym =self.sym_factor if self.epoch>= self.epoch_activate_sym_loss else 0.
        sim_factor = 1.
        loss = sim_factor*sim_loss + factor_sym * sym_reg_loss + factor_scale * scale_reg_loss
        print_out_every_iter = (10* self.step) if self.epoch> self.epoch_activate_multi_step else 10
        if self.iter_count%print_out_every_iter==0:
            if self.epoch >= self.epoch_activate_sym:
                print('sim_loss:{}, factor_sym: {}, sym_reg_loss: {}, factor_scale {}, scale_reg_loss: {}'.format(
                    sim_loss.item(),factor_sym,sym_reg_loss.item(),factor_scale,scale_reg_loss.item())
                )
            else:
                print('sim_loss:{}, factor_scale {}, scale_reg_loss: {}'.format(
                    sim_loss.item(), factor_scale, scale_reg_loss.item())
                )
        return loss


    def get_loss(self):
        """
        :return: the overall loss
        """
        return self.loss



    def forward(self,moving, target,moving_mask=None, target_mask=None):
        """
        forward the affine network

        :param moving: moving image
        :param target: target image
        :return: warped image (intensity[-1,1]), transformation map (coord [-1,1]), affine param
        """
        self.iter_count += 1
        if self.epoch_activate_multi_step>0:
            if self.epoch >= self.epoch_activate_multi_step:
                if self.step_record != self.step:
                    print(" the multi step in affine network activated, multi step num: {}".format(self.step_record))
                self.step = self.step_record
            else:
                self.step = 1
        if self.epoch < self.epoch_activate_sym:
            return self.multi_step_forward(moving, target,moving_mask, target_mask,compute_loss= self.compute_loss)
        else:
            return self.sym_multi_step_forward(moving, target,moving_mask, target_mask)





    def multi_step_forward(self,moving,target, moving_mask=None, target_mask=None, compute_loss=True):
        """
        mutli-step forward, A_t is composed of A_update and A_last

        :param moving: the moving image
        :param target: the target image
        :param compute_loss: if true, compute the loss
        :return: warped image (intensity[-1,1]), transformation map (coord [-1,1]), affine param
        """

        output = None
        moving_cp = moving
        affine_param = None
        affine_param_last = None
        affine_map = None
        bilinear = [Bilinear(self.zero_boundary) for i in range(self.step)]
        self.loss = 0.
        for i in range(self.step):
            #affine_param = self.affine_gen(moving, target)
            if i == 0:
                affine_param = self.affine_gen(moving, target)
            else:
                affine_param = checkpoint(self.affine_gen, moving, target)
            if i > 0:
                affine_param = self.update_affine_param(affine_param, affine_param_last)
            affine_param_last = affine_param
            affine_map = self.gen_affine_map(affine_param)
            output = bilinear[i](moving_cp, affine_map)
            moving = output
            self.affine_param = affine_param
            if compute_loss and (i==self.step-1 or self.acc_multi_step_loss):
                self.loss +=self.compute_overall_loss(self.extern_loss,output, target,affine_map,moving_mask,target_mask)
        if compute_loss and self.acc_multi_step_loss:
            self.loss = self.loss / self.step
        return output, affine_map, affine_param


    def sym_multi_step_forward(self, moving, target,moving_mask=None, target_mask=None):
        """
        symmetry forward
        the "source" is concatenated by source and target, the "target" is concatenated by target and source
        the the multi step foward is called

        :param moving:
        :param target:
        :return:
        """
        self.n_batch = moving.shape[0]
        moving_sym = torch.cat((moving, target), 0)
        target_sym = torch.cat((target, moving), 0)
        moving_mask_sym, target_mask_sym = None, None
        if moving_mask is not None and target_mask is not None:
            moving_mask_sym = torch.cat((moving_mask,target_mask),0)
            target_mask_sym = torch.cat((target_mask,moving_mask),0)
        output, affine_map, affine_param = self.multi_step_forward(moving_sym, target_sym, moving_mask_sym, target_mask_sym)
        return output[:self.n_batch],affine_map[:self.n_batch], affine_param[:self.n_batch]
    def get_extra_to_plot(self):
        """
        no extra image need to be ploted

        :return:
        """
        return None, None



