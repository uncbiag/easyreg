
from .base_model import BaseModel


from  .zhenlin_net import *
from . import networks
from model_pool.model_assamble import Assemble_Net_Test
class Asm_test(BaseModel):
    def name(self):
        return '3D-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)


        ################ settings for default assemble net ############
        network_name =opt['tsk_set']['network_name']
        model_folder_path = opt['tsk_set']['model_folder_path']
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        epoch_list = opt['tsk_set']['model_epoch_list']
        gpu_switcher = (old_gpu_id,cur_gpu_id)
        pars ={'sched':'default','model_name':network_name,'model_folder_path':model_folder_path, 'setting':
            {'in_channel':self.n_in_channel, 'n_class':self.n_class, 'gpu_switcher':gpu_switcher,'epoch_list':epoch_list}}


        # ################ settings for assigned assemble net ############
        # # dic(sched, model_name_list,model_path_list,setting_dic_list)
        #
        # network_name_list =[]
        # model_path_list = []
        # setting_dic_list = [{'in_channel':4, 'n_class':self.n_class, 'gpu_switcher':(0,0)}]
        # pars = {'sched': 'assign', 'model_name_list': network_name_list, 'setting_dic_list':
        #     {'in_channel': n_in_channel, 'n_class': self.n_class, 'gpu_switcher': gpu_switcher}}




        self.network = Assemble_Net_Test(pars)
        self.optimizer, self.lr_scheduler, self.exp_lr_scheduler =self.init_optim(opt['tsk_set']['optim'], self.network)


        self.training_eval_record={}
        print('---------- Networks initialized -------------')
        networks.print_network(self.network)

        if self.isTrain:
            networks.print_network(self.network)
        print('-----------------------------------------------')





    def set_input(self, input, is_train=True):
        self. is_train = is_train
        if is_train:
            self.input = Variable(input[0]['image']).cuda()
        else:
            self.input = Variable(input[0]['image'],volatile=True).cuda()
        self.gt = Variable(input[0]['label']).long().cuda()
        self.fname_list = list(input[1])


    def forward(self,input):
        # here input should be Tensor, not Variable
        return self.network.forward(input)






    def optimize_parameters(self):
        self.iter_count+=1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        output = self.forward(self.input)
        if isinstance(output, list):
            self.output = output[-1]
            self.loss = self.cal_seq_loss(output)
        else:
            self.output = output
            self.loss = self.cal_loss()
        self.backward_net()
        if self.iter_count % self.criticUpdates==0:
            self.optimizer.step()
            self.optimizer.zero_grad()

