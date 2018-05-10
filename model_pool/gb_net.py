
from .base_model import BaseModel

from  .gb_net_pool import *
from . import networks

from model_pool.utils import *


class GBnet(BaseModel):
    def name(self):
        return '3D-GBNet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        network_name =opt['tsk_set']['network_name']
        from .base_model import get_from_model_pool
        #self.network = get_from_model_pool(network_name, self.n_in_channel, self.n_class)
        auto_context = False
        bias= False
        BN = False
        adaboost= True
        ac_mask = int(auto_context)
        model_list = [UNet_light5(self.n_in_channel,self.n_class,bias=bias,BN=BN)]+[UNet_light5(self.n_in_channel+ac_mask*self.n_class,self.n_class,bias=bias,BN=BN) for _ in range(3)]
        self.num_models = len(model_list)
        self.network = gbNet(model_list,self.n_class, end2end=False, auto_context=auto_context, residual=False, adaboost=adaboost)
        self.is_train = opt['tsk_set']['train']

        gbnet_model_s = opt['tsk_set'][('gbnet_model_s', 0, 'start id of the model')]
        gbnet_model_e = opt['tsk_set'][('gbnet_model_e', self.num_models, 'end id of the model')]
        self.cur_model_id = gbnet_model_s if self.is_train else gbnet_model_e-1
        if self.is_train:
            self.network.set_cascaded_train(self.cur_model_id)
        else:
            self.network.set_cascaded_eval()
        #self.network.apply(unet_weights_init)
        self.opt_optim =opt['tsk_set']['optim']
        self.init_optimize_instance(warmming_up=True)
        self.training_eval_record={}
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)


    def set_train(self):
        self.network.training= True
        self.network.debugging = False
        print("cur model_id is {}".format(self.network.cur_model_id))
    def set_val(self):
        self.network.training = False
        self.network.debugging = False
        print("cur model_id is {}".format(self.network.cur_model_id))

    def set_debug(self):
        self.network.debugging = True
        self.network.training = False
        print("cur model_id is {}".format(self.network.cur_model_id))



    def init_optimize_instance(self, warmming_up=False):
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler = self.init_optim(self.opt_optim,self.network,
                                                                                   warmming_up=warmming_up)

    def adjust_learning_rate(self,new_lr=-1):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if new_lr<0:
            lr = self.opt_optim['lr']
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(" no warming up the learning rate is {}".format(lr))


    def set_input(self, input, is_train=True):
        self. is_train = is_train
        if is_train:
            if not self.add_resampled:
                self.input = Variable(input[0]['image'],volatile=True).cuda()
            else:
                self.input =Variable(torch.cat((input[0]['image'], input[0]['resampled_img']),1),volatile=True).cuda()

        else:
            self.input = Variable(input[0]['image'],volatile=True).cuda()
            if 'resampled_img' in input[0]:
                self.resam = Variable( input[0]['resampled_img']).cuda().volatile
        self.gt = Variable(input[0]['label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input, gt = None):
        # here input should be Tensor, not Variable
        return self.network.forward(input, gt)


    def check_and_update_model(self):
        self.cur_model_id += 1
        print("network updated, current model id is {}".format(self.cur_model_id))
        if self.cur_model_id < self.num_models:
            self.network.set_cascaded_train(self.cur_model_id)
        else:
            stop_train = True
            return stop_train







    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        output, weights = self.forward(self.input, self.gt)
        self.output = output
        self.loss = self.loss_fn.get_loss(output,self.gt, weights)
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()

