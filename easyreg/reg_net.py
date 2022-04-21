from .base_mermaid import MermaidBase
from .affine_net import *
from .net_utils import print_network
from .losses import Loss
import torch.optim.lr_scheduler as lr_scheduler
from .utils import *
from .mermaid_net import MermaidNet
from .voxel_morph import VoxelMorphCVPR2018, VoxelMorphMICCAI2019
# from .multiscale_net_new2_opt import Multiscale_FlowNet
#from .lin_unpublic_net import model
from .brainstorm import TransformCVPR2019, AppearanceCVPR2019

model_pool = {
    'affine_sym': AffineNetSym,
    'mermaid': MermaidNet,
    'vm_cvpr': VoxelMorphCVPR2018,
    'vm_miccai': VoxelMorphMICCAI2019,
    # 'multiscale_net':Multiscale_FlowNet,
    #"unpublic_lin": model,
    "bs_trans": TransformCVPR2019,
    'bs_ap': AppearanceCVPR2019
}


class RegNet(MermaidBase):
    """registration network class"""

    def name(self):
        return 'reg-net'

    def initialize(self, opt):
        """
        initialize variable settings of RegNet

        :param opt: ParameterDict, task settings
        :return:
        """
        MermaidBase.initialize(self, opt)
        input_img_sz = opt['dataset']['img_after_resize']
        self.input_img_sz = input_img_sz
        """ the input image sz of the network"""
        spacing_to_refer = opt['dataset'][('spacing_to_refer',[1, 1, 1],'the physical spacing in numpy coordinate')]
        self.spacing = normalize_spacing(spacing_to_refer, self.input_img_sz) if self.use_physical_coord else 1. / (
                    np.array(input_img_sz) - 1)
        """ image spacing"""
        method_name = opt['tsk_set']['method_name']
        self.method_name = method_name
        """the name of the method"""
        self.affine_on = True if 'affine' in method_name else False
        """ perform affine registrtion, if affine is in the network name"""
        self.nonp_on = not self.affine_on
        """ perform affine and nonparametric registration, if mermaid is in the network name"""
        self.network = model_pool[method_name](input_img_sz, opt)
        """create network model"""
        # self.network.apply(weights_init)
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        """update the gradient every # iter"""
        loss_fn = Loss(opt)
        self.network.set_loss_fn(loss_fn)
        self.opt_optim = opt['tsk_set']['optim']
        """settings for the optimizer"""
        self.init_optimize_instance(warmming_up=True)
        """initialize the optimizer and scheduler"""
        self.step_count = 0.
        """ count of the step"""
        self.use_01 = False
        """ the map is normalized to [-1,1] in registration net, todo normalized into [0,1], to be consisitent with mermaid """
        print('---------- Networks initialized -------------')
        print_network(self.network)
        print('-----------------------------------------------')

    def init_optimize_instance(self, warmming_up=False):
        """ get optimizer and scheduler instance"""
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim, self.network,
                                                                                   warmming_up=warmming_up)

    def update_learning_rate(self, new_lr=-1):
        """
        set new learning rate

        :param new_lr: new learning rate
        :return:
        """
        if new_lr < 0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr_scheduler.base_lrs=[lr]
        self.lr_scheduler.last_epoch = 1
        print(" the learning rate now is set to {}".format(lr))

    def set_input(self, data, is_train=True):
        """

        :param data:
        :param is_train:
        :return:
        """
        img_and_label, self.fname_list = data
        self.pair_path = data[0]['pair_path']
        if self.gpu_ids is not None and self.gpu_ids>=0:
            img_and_label['image'] = img_and_label['image'].cuda()
            if 'label' in img_and_label:
                img_and_label['label'] = img_and_label['label'].cuda()
        moving, target, l_moving, l_target = get_reg_pair(img_and_label)
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.original_spacing = data[0]['original_spacing']
        self.original_im_sz =  data[0]['original_sz']


    def init_optim(self, opt, network, warmming_up=False):
        """
        set optimizers and scheduler

        :param opt: settings on optimizer
        :param network: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = opt['optim_type']
        lr = opt['lr']
        beta = opt['adam']['beta']
        lr_sched_opt = opt[('lr_scheduler',{},"settings for learning scheduler")]
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name == 'adam':
            re_optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            re_optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        re_optimizer.zero_grad()
        re_lr_scheduler = None
        re_exp_lr_scheduler = None
        if self.lr_sched_type == 'custom':
            step_size = lr_sched_opt['custom'][('step_size',50,"update the learning rate every # epoch")]
            gamma = lr_sched_opt['custom'][('gamma',0.5,"the factor for updateing the learning rate")]
            re_lr_scheduler = torch.optim.lr_scheduler.StepLR(re_optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            re_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(re_optimizer, mode='min', patience=patience,
                                                                 factor=factor, verbose=True,
                                                                 threshold=threshold, min_lr=min_lr)
        if not warmming_up:
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = opt['lr']/10
            for param_group in re_optimizer.param_groups:
                param_group['lr'] = lr
            re_lr_scheduler.base_lrs = [lr]
            print(" warming up on the learning rate is {}".format(lr))
        return re_optimizer, re_lr_scheduler, re_exp_lr_scheduler

    def cal_loss(self, output=None):
        loss = self.network.get_loss()
        return loss

    def backward_net(self, loss):
        loss.backward()

    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {'file_name': self.fname_list}
        return info

    def forward(self, input=None):
        """

        :param input(not used )
        :return: warped image intensity with [-1,1], transformation map defined in [-1,1], affine image if nonparameteric reg else affine parameter
        """
        if hasattr(self.network, 'set_cur_epoch'):
            self.network.set_cur_epoch(self.cur_epoch)
        output, phi, afimg_or_afparam = self.network.forward(self.moving, self.target, self.l_moving, self.l_target)
        loss = self.cal_loss()
        if not self.is_train and (self.affine_on or self.method_name=="mermaid"):
            save_affine_param_with_easyreg_custom(self.network.affine_param,self.record_path,self.fname_list)

        return output, phi, afimg_or_afparam, loss



    def update_scheduler(self,epoch):
        if self.lr_scheduler is not None and epoch>0:
            self.lr_scheduler.step(epoch)

        for param_group in self.optimizer.param_groups:
            print("the current epoch is {} with learining rate set at {}".format(epoch,param_group['lr']))

    def optimize_parameters(self, input=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate

        :param input: input(not used
        :return:
        """
        if self.is_train:
            self.iter_count += 1
        self.output, self.phi, self.afimg_or_afparam, loss = self.forward()

        self.backward_net(loss / self.criticUpdates)
        self.loss = loss.item()
        if self.iter_count % self.criticUpdates == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        update_lr, lr = self.network.check_if_update_lr()
        if update_lr:
            self.update_learning_rate(lr)

    def do_some_clean(self):
        self.loss = None
        self.gt = None
        self.input = None
        self.output = None
        self.phi = None
        self.afimg_or_afparam = None

    def get_current_errors(self):
        return self.loss

    def get_jacobi_val(self):
        """
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels

        """
        return self.jacobi_val

    def save_image_into_original_sz_with_given_reference(self):
        """
        save the image into original image sz and physical coordinate, the path of reference image should be given

        :return:
        """

        self._save_image_into_original_sz_with_given_reference(self.pair_path, self.phi, inverse_phis=self.inverse_phi,
                                                               use_01=self.use_01)

    def get_extra_to_plot(self):
        """
        extra image to be visualized

        :return: image (BxCxXxYxZ), name
        """
        return self.network.get_extra_to_plot()

    def set_train(self):
        self.network.train(True)
        self.is_train = True
        torch.set_grad_enabled(True)

    def set_val(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_debug(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_test(self):
        self.network.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)
