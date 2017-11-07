from time import time
import torch
from torch.autograd import Variable
from pipLine.utils import *
from models.networks import SimpleNet

record_path ='../data/records/'


def get_criterion(sched):
    if sched == 'L1-loss':
         sched_sel = torch.nn.L1Loss()
    elif sched == "L2-loss":
         sched_sel = torch.nn.MSELoss()
    elif sched == "W-GAN":
        pass
    else:
        raise ValueError(' the criterion is not implemented')
    return sched_sel



def train_model(model, dataloaders, criterion_sched, optimizer, scheduler, num_epochs=25):
    since = time()

    best_model_wts = model.state_dict()
    best_loss = 0.0
    model = model.cuda()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                moving, target = get_pair(data['image'], pair= True)
                input = organize_data(moving, target, sched='depth_concat')

                # wrap them in Variable, remember to optimize this part, the cuda Variable should be warped in dataloader
                moving= Variable(moving.cuda())
                input = Variable(input.cuda())
                target = Variable(target.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(input, moving)
                #_, preds = torch.max(outputs.data, 1)
                criterion = get_criterion(criterion_sched)
                loss = criterion(output, target)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

            epoch_loss = running_loss / dataloaders['data_size'][phase]
            if epoch%10 ==0:
                appendix = 'epoch_'+str(epoch) + '_'+phase
                save_result(record_path, appendix, moving, target, output)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
