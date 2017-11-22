import torch
import os
from models.networks import *
from pipLine.prepare_data import DataManager
from pipLine.train_model import train_model
from pipLine.test_model import test_model
from pipLine.iter_model import iter_model
from tensorboardX import SummaryWriter

logdir= "../data/log"
experiment_name= 'registration_net'
criterion_sched='NCC-loss'
check_point_path = '../data/checkpoints'
record_path ='../data/records/'

path = {}
path['logdir'] = logdir
path['check_point_path'] = check_point_path
path['record_path'] = record_path


def setting_env():
    for item in path:
        if not os.path.exists(item):
            os.mkdir(item)



def pipline(prepare_data, mod, sched):
    dataManager= DataManager('intra')
    if prepare_data:
        dataManager.prepare_data()
    dataloaders = dataManager.dataloaders(batch_size=20)
    model = SimpleNet(dataloaders['info'])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    writer = SummaryWriter(logdir, experiment_name)
    if mod == 'learning':
        if sched == 'train':
            model = train_model(model, dataloaders, criterion_sched=criterion_sched, optimizer=optimizer,
                                scheduler=exp_lr_scheduler,writer= writer, num_epochs=25)
        else:
            state_dic = torch.load(check_point_path + '/model_best.pth.tar')
            model.load_state_dict(state_dic['state_dict'])
        test_model(model, dataloaders,criterion_sched=criterion_sched)
    elif mod == 'itering':
        iter_model(model, dataloaders, criterion_sched=criterion_sched, optimizer=optimizer,
                    scheduler=exp_lr_scheduler, writer= writer, n_iter =100, rel_eps= 1e-5)

    print('Done')



if __name__ == "__main__":
    setting_env()
    pipline(prepare_data=True, mod='learning',sched='train')

