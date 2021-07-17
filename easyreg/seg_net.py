from .base_seg_model import SegModelBase
from .net_utils import print_network
from .losses import Loss
import torch.optim.lr_scheduler as lr_scheduler
from .utils import *
from .seg_unet import SegUnet
from .metrics import get_multi_metric

model_pool = {
    'seg_unet': SegUnet,
}


class SegNet(SegModelBase):
    """segmentation network class"""

    def name(self):
        return 'seg-net'

    def initialize(self, opt):
        """
        initialize variable settings of RegNet

        :param opt: ParameterDict, task settings
        :return:
        """
        SegModelBase.initialize(self, opt)
        method_name = opt['tsk_set']['method_name']
        self.network = model_pool[method_name](opt)
        """create network model"""
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        loss_fn = Loss(opt)
        self.network.set_loss_fn(loss_fn)
        self.opt_optim = opt['tsk_set']['optim']
        """settings for the optimizer"""
        self.init_optimize_instance(warmming_up=True)
        """initialize the optimizer and scheduler"""
        self.step_count = 0.
        """ count of the step"""
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
        print(" the learning rate now is set to {}".format(lr))

    def set_input(self, data, is_train=True):
        """

        :param data:
        :param is_train:
        :return:
        """
        img_and_label, self.fname_list = data
        self.img_path = data[0]['img_path']
        if self.gpu_ids is not None and self.gpu_ids>=0:
            img_and_label['image'] = img_and_label['image'].cuda()
            if 'label' in img_and_label:
                img_and_label['label'] = img_and_label['label'].cuda()
        input, gt = get_seg_pair(img_and_label, is_train)
        self.input = input
        self.input_img_sz =  data[0]['image_after_resize']
        self.gt = gt
        self.spacing = data[0]['original_spacing']

    def init_optim(self, opt, network, warmming_up=False):
        """
        set optimizers and scheduler

        :param opt: settings on optimizer
        :param network: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = opt['optim_type']
        if not warmming_up:
            lr = opt['lr']
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = opt['lr']/10
            print(" warming up on the learning rate is {}".format(lr))
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
        return re_optimizer, re_lr_scheduler, re_exp_lr_scheduler

    def cal_loss(self, output=None, gt=None):
        loss = self.network.get_loss(output, gt)
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
        output = self.network.forward(self.input, self.is_train)
        loss = self.cal_loss(output, self.gt)

        return output, loss

    def update_scheduler(self,epoch):
        if self.lr_scheduler is not None:
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
        self.output, loss = self.forward()

        self.backward_net(loss / self.criticUpdates)
        self.loss = loss.item()
        update_lr, lr = self.network.check_if_update_lr()
        if update_lr:
            self.update_learning_rate(lr)
        if self.iter_count % self.criticUpdates == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()




    def get_current_errors(self):
        return self.loss



    def save_image_into_original_sz_with_given_reference(self):
        """
        save the image into original image sz and physical coordinate, the path of reference image should be given

        :return:
        """
        pass


    def get_evaluation(self):
        sz =self.input_img_sz.squeeze().cpu().numpy().tolist()
        if hasattr(self.network, 'set_file_path'):
            self.network.set_file_path(self.img_path,self.fname_list)

        self.network.set_img_sz(sz)
        output_np = self.network.forward(self.input,self.is_train)
        if self.gt is not None:
            self.val_res_dic = get_multi_metric(output_np, self.gt, rm_bg=False)
        self.output = output_np

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
