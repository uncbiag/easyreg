from time import time
from pipLine.utils import *




def test_model(opt,model, dataloaders):
    since = time()
    model_path = opt['tsk_set']['path']['model_load_path']
    model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on',True,'saving fig')]

    get_test_model(model_path, model.network, model.optimizer)
    running_test_loss=0
    for data in dataloaders['test']:
        # get the inputs
        is_train =  False
        model.network.train(False)
        model.set_input(data, is_train)
        model.cal_test_errors()
        if save_fig_on:
            model.save_fig('best')
        loss = model.get_test_res()
        running_test_loss += loss
    test_loss = running_test_loss /dataloaders['data_size']['test']
    print('the average test_loss: {:.4f}'.format(test_loss))
    time_elapsed = time() - since
    print('the size of test is {}, test evaluation complete in {:.0f}m {:.0f}s'.format( dataloaders['data_size']['test'],
        time_elapsed // 60, time_elapsed % 60))

    return model
