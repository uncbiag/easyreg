from __future__ import print_function

import os
import data_pre.module_parameters as pars
import multiprocessing as mp

# first define all the configuration filenames
this_directory = os.path.dirname(__file__)
# __file__ is the absolute path to the current python file.

task_settings_filename = os.path.join(this_directory, r'../settings/base_task_settings_for_reg.json')
task_settings_filename_comments = os.path.join(this_directory, r'../settings/base_task_settings_for_reg_comments.json')

datapro_settings_filename = os.path.join(this_directory, r'../settings/base_data_settings_for_reg.json')
datapro_settings_filename_comments = os.path.join(this_directory, r'../settings/base_data_settings_for_reg_comments.json')




# noinspection PyStatementEffect
def get_task_settings( task_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    task_params = pars.ParameterDict()

    if task_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        task_settings_filename = os.path.join(this_directory, r'../settings/task_settings.json')

    task_params.load_JSON( task_settings_filename )
    task_params[('tsk_set',{},'settings for task')]
    task_params['tsk_set'][('batch_sz',4,'batch size')]
    task_params['tsk_set'][('task_name','debugging','task_name')]
    task_params['tsk_set'][('train',True,'if training')]
    task_params['tsk_set'][('model','unet','model name, currently only support unet')]
    task_params['tsk_set'][('print_val_detail',False,'print details of validation results')]
    task_params['tsk_set'][('epoch',100,'num of epoch')]
    task_params['tsk_set'][('criticUpdates',1,'criticUpdates')]
    task_params['tsk_set'][('print_step',10,'num of steps to print')]
    task_params['tsk_set'][('old_gpu_ids',0,'the gpu id of the loaded model')]
    task_params['tsk_set'][('gpu_ids',0,'gpu id, currently not support data parallel')]
    task_params['tsk_set'][('continue_train',False,'continue to train')]
    task_params['tsk_set'][('continue_train_lr',5e-5,'set the learning rate when continue the train')]
    task_params['tsk_set'][('model_path','','if continue_train, given the model path')]
    task_params['tsk_set'][('which_epoch','','if continue_train, given the epoch')]
    task_params['tsk_set'][('max_batch_num_per_epoch',[200,8,4],'number of pairs per training/val/debug epoch,  [200,8,5] refers to 200 pairs for each train epoch, 8 pairs for each validation epoch and 5 pairs for each debug epoch')]
    task_params['tsk_set'][('check_best_model_period',5,'num of epoch to check the best model')]
    task_params['tsk_set'][('save_val_fig_epoch',5,'epoch to save val fig')]
    task_params['tsk_set'][('save_by_standard_label',True,'save the label in original label index, for example if the original label is a way like [ 1,3,7,8], otherwise save in [0,1,2,3]')]
    task_params['tsk_set'][('n_in_channel',1,'number of input channel')]
    task_params['tsk_set'][('image size',-1,'provided in the datapro_params')]
    task_params['tsk_set'][('input_resize_factor',-1,'provided in the datapro_params')]
    task_params['tsk_set'][('val_period',10,'do validation every # epoch')]


    task_params['tsk_set'][('optim',{},'settings for adam')]
    task_params['tsk_set']['optim'][('optim_type','adam','settings for adam')]
    task_params['tsk_set']['optim'][('lr',0.001,'learning rate')]
    task_params['tsk_set']['optim'][('adam',{},'settings for adam')]
    task_params['tsk_set']['optim']['adam'][('beta',0.9,'settings for adam')]
    task_params['tsk_set']['optim'][('lr_scheduler',{},'settings for lr_scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler'][('type','custom','steps to decay learning rate')]
    task_params['tsk_set']['optim']['lr_scheduler'][('plateau',{},'settings fort plateau scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler']['plateau'][('patience',20,'settings fort plateau scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler']['plateau'][('factor',0.2,'settings fort plateau scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler']['plateau'][('threshold',0.001,'settings fort plateau scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler']['plateau'][('min_lr',1e-6,'settings fort plateau scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler'][('custom',{},'settings for custom scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler']['custom'][('step_size',4000*3,'steps to decay learning rate')]
    task_params['tsk_set']['optim']['lr_scheduler']['custom'][('gamma',0.5,'factor to decay learning rate')]


    task_params['tsk_set'][('loss', {}, 'settings for adam')]
    task_params['tsk_set']['loss'][('type','ce','loss name, {ce,mse,l1_loss,focal_loss, dice_loss}')]


    task_params['tsk_set']['network_name','affine_cycle','method supported by the model, see the guideline']

    task_params['tsk_set'][('reg',{},"settings for registration task")]
    task_params['tsk_set']['reg'][('low_res_factor', 0.5,'low resolution map factor for non-parametric method')]
    task_params['tsk_set']['reg'][('mermaid_net',{},"settings for mermaid net")]
    task_params['tsk_set']['reg']['mermaid_net'][('using_sym',False,'using symmetric training, if true, the loss is combined with source2target, target2source and symmetric loss')]
    task_params['tsk_set']['reg']['mermaid_net'][('sym_factor', 100,'the weight for the symmetric factor')]
    task_params['tsk_set']['reg']['mermaid_net'][('using_complex_net', True, 'using a complex unet if True')]
    task_params['tsk_set']['reg']['mermaid_net'][('using_multi_step', True,'using multi-step training for mermaid_based method')]
    task_params['tsk_set']['reg']['mermaid_net'][('num_step',1,'number of steps in multi-step mermaid based method')]
    task_params['tsk_set']['reg']['mermaid_net'][('using_lddmm',False,'True: using lddmm (copyright in mermaid) Flase: using vSVF (copyright in mermaid)')]
    task_params['tsk_set']['reg']['mermaid_net'][('using_affine_init', True, 'using affine network ( need to be trained first) for initailization False: using id transform as initialization')]
    affine_path = '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_affine_net_sym_lncc/checkpoints/epoch_1070_'
    task_params['tsk_set']['reg']['mermaid_net'][('affine_init_path',affine_path,' if using_affine_init=True, the path of the affine model should be provided')]
    task_params['tsk_set']['reg'][('affine_net',{},'settings for multi-step affine network')]
    task_params['tsk_set']['reg']['affine_net'][('affine_net_iter',1,'number of steps used')]
    task_params['tsk_set']['reg']['affine_net'][('using_complex_net', True, 'using complex version of affine network')]

    task_params['tsk_set']['reg'][('mermaid_iter',{},'settings for optimization-based mermaid iteration')]
    task_params['tsk_set']['reg']['mermaid_iter'][('affine',{},'settings for affine in mermaid iteration')]
    task_params['tsk_set']['reg']['mermaid_iter']['affine'][('sigma',0.7,' loss = regterm + 1/(sigma)^2*simterm, recommand np.sqrt(batch_sz/4) for longitudinal, recommand np.sqrt(batch_sz/2) for cross-subject')]


    return task_params



def get_datapro_settings(datapro_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    datapro_params = pars.ParameterDict()

    if datapro_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        datapro_settings_filename = os.path.join(this_directory, r'../settings/base_data_settings_for_reg.json')

    datapro_params.load_JSON( datapro_settings_filename )
    datapro_params[('datapro',{},'settings for the data process')]
    datapro_params['datapro'][('task_type', " 'seg' or 'reg', namely segmentation or registration")]

    datapro_params['datapro'][('dataset', {}, 'general settings for dataset')]
    datapro_params['datapro']['dataset'][('dataset_name', 'lpba', 'name of the dataset: oasis2d, lpba, ibsr, cmuc')]
    datapro_params['datapro']['dataset'][('task_name', 'lpba_affined', 'task name for data process')]
    datapro_params['datapro']['dataset'][('data_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('label_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('output_path', '/playpen/zyshen/data/', "the path to save the processed data")]
    datapro_params['datapro']['dataset'][('prepare_data', False, 'prepare the data ')]
    datapro_params['datapro']['dataset'][('using_normalize', True, 'normalized the data ')]
    datapro_params['datapro']['dataset'][('divided_ratio', (0.7, 0.2, 0.1), 'divided the dataset into train, val and test set by the divided_ratio')]
    datapro_params['datapro']['dataset'][('img_size',[160,384,384], 'image size in numpy coord')]
    datapro_params['datapro']['switch'][('switch_to_exist_task', False, 'switch to existed task without modify other datapro settings')]
    datapro_params['datapro']['switch'][('task_root_path', '/playpen/zyshen/data/oasis_inter_slicing90', 'path of existed processed data')]

    datapro_params['datapro'][('reg', {}, 'general settings for dataset')]
    datapro_params['datapro']['reg'][('sched', 'inter', "['inter'|'intra'], 'intra' for longitudinal,'inter' for cross-subject,  is not used if the data has manually prepared")]
    datapro_params['datapro']['reg'][('all_comb', False, 'all possible pair combination ')]
    datapro_params['datapro']['reg'][('slicing', -1, 'the index to be sliced from the 3d image dataset, support lpba, ibsr, cmuc')]
    datapro_params['datapro']['reg'][('axis', -1, 'which axis needed to be sliced')]
    datapro_params['datapro']['reg'][('input_resize_factor',[80./160.,192./384.,192./384], 'resize the image by [factor_x,factor_y, factor_z]')]
    datapro_params['datapro']['reg'][('max_pair_for_loading', [100,10,30,10], 'limit the max number of the pairs for [train, val, test, debug]')]
    datapro_params['datapro']['reg'][('load_training_data_into_memory',False, 'load all training pairs into memory')]



    return datapro_params







# write out the configuration files (when called as a script; in this way we can boostrap a new configuration)

if __name__ == "__main__":
    task_params = pars.ParameterDict()
    task_params.write_JSON(task_settings_filename)
    task_params = get_task_settings()
    task_params.write_JSON(task_settings_filename)
    task_params.write_JSON_comments(task_settings_filename_comments)

    datapro_params = pars.ParameterDict()
    datapro_params.write_JSON(datapro_settings_filename)
    datapro_params = get_datapro_settings()
    datapro_params.write_JSON(datapro_settings_filename)
    datapro_params.write_JSON_comments(datapro_settings_filename_comments)


    # respro_params = get_respro_settings()
    # respro_params.write_JSON(respro_settings_filename)
    # respro_params.write_JSON_comments(respro_settings_filename_comments)




