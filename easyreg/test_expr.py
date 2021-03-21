from time import time
from .net_utils import get_test_model
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
    cur_gpu_id = opt['tsk_set'][('gpu_ids', -1,"the gpu id")]
    task_type = opt['dataset'][('task_type','reg',"the task type, either 'reg' or 'seg'")]
    running_range=[-1]#opt['tsk_set']['running_range']  # todo should be [-1]
    running_part_data = running_range[0]>=0
    if running_part_data:
        print("running part of the test data from range {}".format(running_range))
    gpu_id = cur_gpu_id

    if model.network is not None and gpu_id>=0:
        model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on', True, 'saving fig')]
    save_running_resolution_3d_img = opt['tsk_set'][('save_running_resolution_3d_img', True, 'saving fig')]
    output_taking_original_image_format = opt['tsk_set'][('output_taking_original_image_format', False, 'output follows the same sz and physical format of the original image (input by command line or txt)')]

    phases = ['test'] #['val','test']  ###################################3
    if len(model_path):
        get_test_model(model_path, model.network,  model.optimizer)     ##############TODO  model.optimizer
    else:
        print("Warning, the model is not manual loaded, make sure your model itself has been inited")

    model.set_cur_epoch(-1)
    for phase in phases:
        num_samples = len(dataloaders[phase])
        if running_part_data:
            num_samples = len(running_range)
        records_score_np = np.zeros(num_samples)
        records_time_np = np.zeros(num_samples)
        if task_type == 'reg':
            records_jacobi_val_np = np.zeros(num_samples)
            records_jacobi_num_np = np.zeros(num_samples)
        loss_detail_list = []
        jacobi_val_res = 0.
        jacobi_num_res = 0.
        running_test_score = 0
        time_total= 0
        for idx, data in enumerate(dataloaders[phase]):
            i= idx
            if running_part_data:
                if i not in running_range:
                    continue
                i = i - running_range[0]

            batch_size = len(data[0]['image'])
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
            if save_running_resolution_3d_img:
                model.save_fig_3D(phase='test')
                if task_type == 'reg':
                    model.save_deformation()

            if output_taking_original_image_format:
                model.save_image_into_original_sz_with_given_reference()


            loss,loss_detail = model.get_test_res(detail=True)
            print("the loss_detailed is {}".format(loss_detail))
            running_test_score += loss * batch_size
            records_score_np[i] = loss
            loss_detail_list += [loss_detail]
            print("id {} and current pair name is : {}".format(i,data[1]))
            print('the current running_score:{}'.format(loss))
            print('the current average running_score:{}'.format(running_test_score/(i+1)/batch_size))
            if task_type == 'reg':
                jaocbi_res = model.get_jacobi_val()
                if jaocbi_res is not None:
                    jacobi_val_res += jaocbi_res[0] * batch_size
                    jacobi_num_res += jaocbi_res[1] * batch_size
                    records_jacobi_val_np[i] = jaocbi_res[0]
                    records_jacobi_num_np[i] = jaocbi_res[1]
                    print('the current jacobi is {}'.format(jaocbi_res))
                    print('the current averge jocobi val is {}'.format(jacobi_val_res/(i+1)/batch_size))
                    print('the current averge jocobi num is {}'.format(jacobi_num_res/(i+1)/batch_size))
        test_score = running_test_score / len(dataloaders[phase].dataset)
        time_per_img = time_total / len((dataloaders[phase].dataset))
        print('the average {}_loss: {:.4f}'.format(phase, test_score))
        print("the average time for per image is {}".format(time_per_img))
        time_elapsed = time() - since
        print('the size of {} is {}, evaluation complete in {:.0f}m {:.0f}s'.format(len(dataloaders[phase].dataset),phase,
                                                                                           time_elapsed // 60,
                                                                                           time_elapsed % 60))
        np.save(os.path.join(record_path,task_name+'records'),records_score_np)
        records_detail_np = extract_interest_loss(loss_detail_list,sample_num=len(dataloaders[phase].dataset))
        np.save(os.path.join(record_path,task_name+'records_detail'),records_detail_np)
        np.save(os.path.join(record_path,task_name+'records_time'),records_time_np)
        if task_type ==  'reg':
            jacobi_val_res = jacobi_val_res / len(dataloaders[phase].dataset)
            jacobi_num_res = jacobi_num_res / len(dataloaders[phase].dataset)
            print("the average {}_ jacobi val: {}  :".format(phase, jacobi_val_res))
            print("the average {}_ jacobi num: {}  :".format(phase, jacobi_num_res))
            np.save(os.path.join(record_path, task_name + 'records_jacobi'), records_jacobi_val_np)
            np.save(os.path.join(record_path, task_name + 'records_jacobi_num'), records_jacobi_num_np)
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


