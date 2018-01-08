import torch
import os
from models.networks import *
from model_pool.initializer import Initializer
from pipLine.train_model_reg import train_model
from pipLine.test_model_reg import test_model
from pipLine.iter_model import iter_model
from model_pool.create_model import create_model
from .train_expr import train_model

logdir= "../data/log"
experiment_name= 'registration_net'
criterion_sched='NCC-loss'
check_point_path = '../data/checkpoints'
record_path ='../data/records/'

path = {}
path['logdir'] = logdir
path['check_point_path'] = check_point_path
path['record_path'] = record_path
model_name = 'SimpleNet'

def selModel(mode_n):
    if mode_n == 'SimpleNet':
        return SimpleNet
    elif mode_n == 'FlowNet':
        return FlowNet
    else:
        raise ValueError('not implemented yet')


def setting_env():
    for item in path:
        if not os.path.exists(path[item]):
            os.mkdir(path[item])




def run_task(task_opt):
    dataManager= DataManager('inter')
    if prepare_data:
        dataManager.prepare_data()
    dataloaders = dataManager.dataloaders(batch_size=10)
    mode_sel = selModel(model_name)
    model = mode_sel(dataloaders['info'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    writer = SummaryWriter(logdir, experiment_name)
    if mod == 'learning':  # learning from big data
        if sched == 'train':
            model = train_model(model, dataloaders, criterion_sched=criterion_sched, optimizer=optimizer,
                                scheduler=exp_lr_scheduler,writer= writer, num_epochs=25, clip_grad=True, experiment_name= experiment_name )
        else:
            state_dic = torch.load(check_point_path + '/model_best.pth.tar')
            model.load_state_dict(state_dic['state_dict'])
        test_model(model, dataloaders,criterion_sched=criterion_sched)
    elif mod == 'itering':  # itering between two image
        iter_model(model, dataloaders, criterion_sched=criterion_sched, optimizer=optimizer,
                    scheduler=exp_lr_scheduler, writer= writer, n_iter =100, rel_eps= 1e-5)

    print('Done')



class pipline():
    def initialize(self):
        initializer = Initializer()
        initializer.initialize_data_manager(task_type='seg')
        self.tsk_opt = initializer.init_task_option()
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.get_data_loader(batch_size =5)
        self.model = create_model(self.tsk_opt)

    def run_task(self):
        if self.tsk_opt['train']:
            train_model(self.tsk_opt, self.model, self.data_loaders,self.writer)
        else:
            text_expr(self.tsk_opt, self.data_loaders)





