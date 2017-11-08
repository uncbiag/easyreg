from time import time
import torch
from torch.autograd import Variable
from pipLine.utils import *
from models.networks import SimpleNet

record_path ='../data/records/'


def test_model(model, dataloaders, criterion_sched):
    since = time()

    model = model.cuda()

    model.train(False)  # Set model to evaluate mode

    iter = 0

    for data in dataloaders['test']:
        # get the inputs
        moving, target = get_pair(data['image'], pair=True)
        input = organize_data(moving, target, sched='depth_concat')
        batch_size = input.size(0)

        # wrap them in Variable, remember to optimize this part, the cuda Variable should be warped in dataloader
        moving= Variable(moving.cuda())
        input = Variable(input.cuda())
        target = Variable(target.cuda())

        output = model(input, moving)
        criterion = get_criterion(criterion_sched)
        loss = criterion(output, target)
        print("average loss:{}".format(loss.cpu().data.numpy()))

        appendix = 'img_' + str(iter)
        save_result(record_path + 'test' + '/', appendix, moving, target, output)
        iter += 1

    time_elapsed = time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
