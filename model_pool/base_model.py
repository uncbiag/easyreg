import os
import torch
from model_pool.utils import *
import torch.optim.lr_scheduler as lr_scheduler

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt['tsk_set']['gpu_ids']
        self.isTrain = opt['tsk_set']['train']
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt['tsk_set']['path']['check_point_path']
        self.record_path = opt['tsk_set']['path']['record_path']
        self.spacing = opt['tsk_set']['extra_info']['spacing']
        self.img_sz = opt['tsk_set']['extra_info']['img_sz']
        self.continue_train = opt['tsk_set']['continue_train']
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.optimizer= None

        self.iter_count = 0
        self.dim = len(self.img_sz)
        self.network =None


    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def init_optim(self, opt):
        optimize_name = opt['optim_type']
        lr = opt['lr']
        beta = opt['adam']['beta']
        lr_sched_opt = opt['lr_scheduler']
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler = None
        self.exp_lr_scheduler = None
        if self.lr_sched_type == 'custom':
            step_size = lr_sched_opt['custom']['step_size']
            gamma = lr_sched_opt['custom']['gamma']
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            self.exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience,
                                                                   factor=factor, verbose=True,
                                                                   threshold=threshold, min_lr=min_lr)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def save_2d_visualize(self,input,gt,output):
        image_summary = make_image_summary(input, gt, output)



