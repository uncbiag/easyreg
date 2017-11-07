import torch
from models.networks import *
from pipLine.prepare_data import DataManager
from pipLine.train import train_model


def pipline():
    dataManager= DataManager('intra')
    dataManager.prepare_data()
    dataloaders = dataManager.dataloaders(batch_size=20)
    model = SimpleNet(dataloaders['info'])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_conv = train_model(model, dataloaders, criterion_sched="L1-loss", optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=25)


if __name__ == "__main__":
    pipline()

