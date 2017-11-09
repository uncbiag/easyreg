from time import time
import torch
from torch.autograd import Variable
from pipLine.utils import *
from models.networks import SimpleNet

record_path ='../data/records/'
model_path = None
check_point_path = '../data/checkpoints'
reg= 1e-4



def train_model(model, dataloaders, criterion_sched, optimizer, scheduler,writer, num_epochs=25):
    since = time()

    best_model_wts = model.state_dict()
    best_loss = 0
    model = model.cuda()
    start_epoch = 0
    global_step = {x:0 for x in ['train','val']}
    period_loss = {x: 0. for x in ['train', 'val']}
    period =20
    if model_path is not None:
        start_epoch, best_loss =resume_train(model_path, model)
    # else:
    #     model.apply(weights_init)


    for epoch in range(start_epoch, num_epochs):
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

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                moving, target = get_pair(data['image'], pair= True)
                input = organize_data(moving, target, sched='depth_concat')
                batch_size = input.size(0)

                # wrap them in Variable, remember to optimize this part, the cuda Variable should be warped in dataloader
                moving= Variable(moving.cuda())
                input = Variable(input.cuda())
                target = Variable(target.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output, gradField = model(input, moving)
                #_, preds = torch.max(outputs.data, 1)
                criterion = get_criterion(criterion_sched)
                loss = criterion(output, target)
                loss += reg * torch.sum(gradField)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                period_loss[phase] += loss.data[0]
                # save for tensorboard, both train and val will be saved
                global_step[phase] += 1
                if global_step[phase] > 1 and global_step[phase]%period == 0:
                    period_avg_loss = period_loss[phase] / period
                    writer.add_scalar('loss/'+phase, period_avg_loss, global_step[phase])
                    period_loss[phase] = 0.



            epoch_loss = running_loss / dataloaders['data_size'][phase]
            if epoch%10 == 0:
                appendix = 'epoch_'+str(epoch)
                save_result(record_path+phase+'/', appendix, moving, target, output)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if epoch == 0:
                best_loss = epoch_loss
            is_best =False
            if phase == 'val' and epoch_loss < best_loss:
                is_best = True
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
            # save check point every epoch
            # only train phase would be saved
            if phase == 'val':
                save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),
                             'best_loss': best_loss}, is_best, check_point_path, 'epoch_'+str(epoch), 'reg_net')


        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
