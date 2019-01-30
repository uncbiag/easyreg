from time import time
from model_pool.net_utils import get_test_model
import os
import numpy as np



def test_model(opt,model, dataloaders):

    model_path = opt['tsk_set']['path']['model_load_path']
    if isinstance(model_path, list):
        for i, path in enumerate(model_path):
            __test_model(opt,model,dataloaders,path,str(i)+'_')
    else:
        __test_model(opt,model, dataloaders,model_path)




def __test_model(opt,model,dataloaders, model_path,task_name=''):
    since = time()
    record_path = opt['tsk_set']['path']['record_path']
    label_num = opt['tsk_set']['extra_info']['num_label']

    if model.network is not None:
        model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on', True, 'saving fig')]

    phases = ['test'] #['val','test']  ###################################3
    if len(model_path):
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        get_test_model(model_path, model.network,  model.optimizer,old_gpu=old_gpu_id,cur_gpu=cur_gpu_id)     ##############TODO  model.optimizer
    else:
        print("Warning, the model is not manual loaded, make sure your model itself has been inited")

    model.set_cur_epoch(-1)
    for phase in phases:
        num_samples = len(dataloaders[phase])
        records_score_np = np.zeros(num_samples)
        records_jacobi_val_np = np.zeros(num_samples)
        records_jacobi_num_np = np.zeros(num_samples)
        records_time_np = np.zeros(num_samples)
        loss_detail_list = []
        jacobi_val_res = 0.
        jacobi_num_res = 0.
        running_test_loss = 0
        time_total= 0
        for i, data in enumerate(dataloaders[phase]):
            batch_size =  len(data[0]['image'])
            is_train = False
            if model.network is not None:
                model.network.train(False)
            model.set_val()
            model.set_input(data, is_train)
            ex_time = time()
            model.cal_test_errors()
            batch_time = time() - ex_time
            time_total += batch_time
            print("the batch sample registration takes {} to complete".format(batch_time))
            records_time_np[i] = batch_time
            if save_fig_on:
                model.save_fig('debug_model_'+phase)
            loss,loss_detail = model.get_test_res(detail=True)
            running_test_loss += loss * batch_size
            extra_res  = model.get_extra_res()
            if extra_res is not None:
                jacobi_val_res += extra_res[0] * batch_size
                jacobi_num_res += extra_res[1] * batch_size
                records_jacobi_val_np[i] = extra_res[0]
                records_jacobi_num_np[i] = extra_res[1]
            records_score_np[i] = loss
            loss_detail_list += [loss_detail]
            print("id {} and current pair name is : {}".format(i,data[1]))
            print('the current running_loss:{}'.format(loss))
            print('the current average running_loss:{}'.format(running_test_loss/(i+1)/batch_size))
            print('the current jocobi is {}'.format(extra_res))
            print('the current averge jocobi val is {}'.format(jacobi_val_res/(i+1)/batch_size))
            print('the current averge jocobi num is {}'.format(jacobi_num_res/(i+1)/batch_size))
        test_loss = running_test_loss / len(dataloaders[phase].dataset)
        jacobi_val_res = jacobi_val_res/len(dataloaders[phase].dataset)
        jacobi_num_res = jacobi_num_res/len(dataloaders[phase].dataset)
        time_per_img = time_total/len((dataloaders[phase].dataset))
        print('the average {}_loss: {:.4f}'.format(phase,test_loss))
        print("the average {}_ jacobi val: {}  :".format(phase, jacobi_val_res))
        print("the average {}_ jacobi num: {}  :".format(phase, jacobi_num_res))
        print("the average time for per image is {}".format(time_per_img))
        time_elapsed = time() - since
        print('the size of {} is {}, evaluation complete in {:.0f}m {:.0f}s'.format(len(dataloaders[phase].dataset),phase,
                                                                                           time_elapsed // 60,
                                                                                           time_elapsed % 60))
        np.save(os.path.join(record_path,task_name+'records'),records_score_np)
        records_detail_np = extract_interest_loss(loss_detail_list,sample_num=len(dataloaders[phase].dataset))
        np.save(os.path.join(record_path,task_name+'records_detail'),records_detail_np)
        np.save(os.path.join(record_path,task_name+'records_jacobi'),records_jacobi_val_np)
        np.save(os.path.join(record_path,task_name+'records_jacobi_num'),records_jacobi_num_np)
        np.save(os.path.join(record_path,task_name+'records_time'),records_time_np)

    return model


def extract_interest_loss(loss_detail_list,sample_num):
    """" multi_metric_res:{iou: Bx #label , dice: Bx#label...} ,"""
    assert len(loss_detail_list)>0
    if isinstance(loss_detail_list[0],dict):
        label_num =  loss_detail_list[0]['dice'].shape[1]
        records_detail_np = np.zeros([sample_num,label_num])
        sample_count = 0
        for multi_metric_res in loss_detail_list:
            batch_len = multi_metric_res['dice'].shape[0]
            records_detail_np[sample_count:sample_count+batch_len,:] = multi_metric_res['dice']
            sample_count += batch_len
    else:
        records_detail_np=np.array([-1])
    return records_detail_np


