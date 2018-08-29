from time import time
from pipLine.utils import *




def debug_model(opt,model, dataloaders):
    since = time()
    model_path = opt['tsk_set']['path']['model_load_path']
    if model.network is not None:
        model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on', True, 'saving fig')]

    phases = ['test'] #['val','test']  ###################################3
    if len(model_path):
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        get_test_model(model_path, model.network, model.optimizer,old_gpu=old_gpu_id,cur_gpu=cur_gpu_id)
    else:
        print("Warning, the model is not manual loaded, make sure your model itself has been inited")

    for phase in phases:
        running_test_loss = 0
        for data in dataloaders[phase]:
            # get the inputs
            is_train = False
            if model.network is not None:
                model.network.train(False)
            model.set_val()
            model.set_input(data, is_train)
            model.cal_test_errors()
            if save_fig_on:
                model.save_fig('debug_model_'+phase)
            loss,_ = model.get_test_res()
            running_test_loss += loss * len(data[0]['image'])
            print('the current running_loss:{}'.format(loss))
        test_loss = running_test_loss / len(dataloaders[phase].dataset)
        print('the average {}_loss: {:.4f}'.format(phase,test_loss))
        time_elapsed = time() - since
        print('the size of {} is {}, evaluation complete in {:.0f}m {:.0f}s'.format(len(dataloaders[phase].dataset),phase,
                                                                                           time_elapsed // 60,
                                                                                           time_elapsed % 60))

    return model
