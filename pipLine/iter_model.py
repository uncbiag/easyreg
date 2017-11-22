from time import time
import torch
from torch.autograd import Variable
from pipLine.utils import *
from models.networks import SimpleNet

record_path ='../data/records/itering/'
model_path = None
check_point_path = '../data/checkpoints'
reg= 1e-4



def iter_model(model, dataloaders, criterion_sched, optimizer, scheduler, writer, n_iter , rel_eps):

    model = model.cuda()
    model.train(True)
    period =20

    for phase in ['train', 'val','test']:   # using the whole dataset
        for idx, data in enumerate(dataloaders[phase]):
            print("processing {} / {} in {} set".format(idx, dataloaders['data_size'][phase], phase))
            since = time()
            moving, target = get_pair(data['image'], pair=True)
            input = organize_data(moving, target, sched='depth_concat')
            batch_size = input.size(0)
            running_loss = 0.0
            # wrap them in Variable, remember to optimize this part, the cuda Variable should be warped in dataloader
            moving = Variable(moving.cuda())
            input = Variable(input.cuda())
            target = Variable(target.cuda())
            for i_iter in range(n_iter):
                if idx ==0:
                    scheduler.step()
                optimizer.zero_grad()
                output, gradField = model(input, moving)
                criterion = get_criterion(criterion_sched)
                loss = criterion(output, target)
                #loss += reg * torch.sum(gradField)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

                if i_iter == n_iter-1:
                    print('iter: {}/{}'.format(i_iter, n_iter))
                    print('the loss: {}'.format(loss.data[0]))
                    appendix = phase + '_it'+ str(i_iter)
                    save_result(record_path + phase + '_id'+ str(idx) + '/', appendix, moving, target, output)

                if i_iter % 10 == 0:
                    iter_rec_loss = running_loss / 10
                    writer.add_scalar('iter_loss/'+phase, iter_rec_loss,i_iter)
                    running_loss = 0.
            print()
            time_elapsed = time() - since
            print('Iter complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

