import torch
from models.networks import *
from pipLine.prepare_data import DataManager
from pipLine.train import train_model
from pipLine.test import test_model
from tensorboardX import SummaryWriter

logdir= "../data/log"
experiment_name= 'registration_net'
criterion_sched='L1-loss'
check_point_path = '../data/checkpoints'


def pipline(sched, prepare_data=True):
    dataManager= DataManager('intra')
    if prepare_data:
        dataManager.prepare_data()
    dataloaders = dataManager.dataloaders(batch_size=20)
    model = SimpleNet(dataloaders['info'])
    if sched == 'train':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        writer = SummaryWriter(logdir, experiment_name)
        model = train_model(model, dataloaders, criterion_sched=criterion_sched, optimizer=optimizer,
                            scheduler=exp_lr_scheduler,writer= writer, num_epochs=25)
    state_dic = torch.load(check_point_path + '/model_best.pth.tar')
    model.load_state_dict(state_dic['state_dict'])
    test_model(model, dataloaders,criterion_sched=criterion_sched)
    print('Done')



if __name__ == "__main__":
    pipline('train',prepare_data=True)

