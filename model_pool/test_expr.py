from time import time
from pipLine.utils import *
from glob import glob



def test_model(opt,model, dataloaders):
    since = time()
    model_path = opt['tsk_set']['path']['model_load_path']
    model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on', True, 'saving fig')]
    dg_key_word = opt['tsk_set']['dg_key_word']
    cur_gpu_id = opt['tsk_set']['gpu_ids']
    old_gpu_id = opt['tsk_set']['old_gpu_ids']

    get_test_model(model_path, model.network, model.optimizer, old_gpu=old_gpu_id, cur_gpu=cur_gpu_id)
    running_test_loss = 0
    for data in dataloaders['test']:
        # get the inputs
        is_train = False
        model.network.train(False)
        model.set_input(data, is_train)
        model.get_pred_img(split_size=3)
        # model.get_output_map()
        if save_fig_on:
            model.save_fig(dg_key_word, standard_record=True, saving_gt=False)



def test_asm_model(opt,model, dataloaders):
    since = time()
    model_path = opt['tsk_set']['path']['check_point_path']
    model.network = model.network.cuda()
    save_fig_on = opt['tsk_set'][('save_fig_on',True,'saving fig')]
    dg_key_word = opt['tsk_set']['dg_key_word']

    running_test_loss=0
    for data in dataloaders['test']:
        # get the inputs
        is_train =  False
        model.network.train(False)
        model.set_input(data, is_train)
        model.get_pred_img(split_size=4)
        # model.get_output_map()
        if save_fig_on:
            model.save_fig(dg_key_word, standard_record=True, saving_gt=False)