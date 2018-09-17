from time import time
from pipLine.utils import *
import os



def debug_model(opt,model, dataloaders):
    since = time()
    model_path = opt['tsk_set']['path']['model_load_path']
    record_path = opt['tsk_set']['path']['record_path']
    label_num = opt['tsk_set']['extra_info']['num_label']

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
        num_samples = len(dataloaders[phase])
        records_np = np.zeros(num_samples)
        loss_detail_list = []
        running_test_loss = 0
        for i, data in enumerate(dataloaders[phase]):
            # get the inputs
            is_train = False
            if model.network is not None:
                model.network.train(False)
            model.set_val()
            model.set_input(data, is_train)
            model.cal_test_errors()
            if save_fig_on:
                model.save_fig('debug_model_'+phase)
            loss,loss_detail = model.get_test_res(detail=True)
            running_test_loss += loss * len(data[0]['image'])
            records_np[i] = loss
            loss_detail_list += [loss_detail]
            print('the current running_loss:{}'.format(loss))
        test_loss = running_test_loss / len(dataloaders[phase].dataset)
        print('the average {}_loss: {:.4f}'.format(phase,test_loss))
        time_elapsed = time() - since
        print('the size of {} is {}, evaluation complete in {:.0f}m {:.0f}s'.format(len(dataloaders[phase].dataset),phase,
                                                                                           time_elapsed // 60,
                                                                                           time_elapsed % 60))
        np.save(os.path.join(record_path,'records'),records_np)
        records_detail_np = extract_interest_loss(loss_detail_list,sample_num=len(dataloaders[phase].dataset),label_num=label_num)
        np.save(os.path.join(record_path,'records_detail'),records_detail_np)


    return model

def extract_interest_loss(loss_detail_list,sample_num, label_num):
    """" multi_metric_res:{iou: Bx #label , dice: Bx#label...} ,"""
    assert len(loss_detail_list)>=0
    records_detail_np = np.zeros([sample_num,label_num])
    sample_count = 0
    for multi_metric_res in loss_detail_list:
        batch_len = multi_metric_res['dice'].shape[0]
        records_detail_np[sample_count:sample_count+batch_len,:] = multi_metric_res['dice']
        sample_count += batch_len
    return records_detail_np


