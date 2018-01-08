from __future__ import print_function

import os
import data_pre.module_parameters as pars
import multiprocessing as mp

# first define all the configuration filenames
this_directory = os.path.dirname(__file__)
# __file__ is the absolute path to the current python file.

task_settings_filename = os.path.join(this_directory, r'../settings/task_settings.json')
task_settings_filename_comments = os.path.join(this_directory, r'../settings/task_settings_comments.json')

datapro_settings_filename = os.path.join(this_directory, r'../settings/data_settings.json')
datapro_settings_filename_comments = os.path.join(this_directory, r'../settings/data_settings_comments.json')


respro_settings_filename = os.path.join(this_directory, r'../settings/respro_settings.json')
respro_settings_filename_comments = os.path.join(this_directory, r'../settings/respro_settings_comments.json')





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
    task_params['tsk_set'][('task_name','','task_name')]
    task_params['tsk_set'][('train',False,'if training')]
    task_params['tsk_set'][('model','unet','model name, currently only support unet')]
    task_params['tsk_set'][('loss','ce','loss name, {ce,mse,l1_loss,focal_loss, dice_loss}')]
    task_params['tsk_set'][('epoch',100,'num of epoch')]
    task_params['tsk_set'][('print_step',10,'num of steps to print')]
    task_params['tsk_set'][('gpu_ids',0,'gpu id, currently not support data parallel')]
    task_params['tsk_set'][('continue_train',False,'continue to train')]
    task_params['tsk_set'][('model_path','','if continue_train, given the model path')]
    task_params['tsk_set'][('which_epoch','','if continue_train, given the epoch')]

    task_params['tsk_set'][('optim',{},'settings for adam')]
    task_params['tsk_set']['optim'][('optimizer','adam','settings for adam')]
    task_params['tsk_set']['optim'][('lr',0.001,'learning rate')]
    task_params['tsk_set']['optim'][('adam',{},'settings for adam')]
    task_params['tsk_set']['optim']['adam'][('beta',0.9,'settings for adam')]
    task_params['tsk_set']['optim'][('lr_scheduler',{},'settings for lr_scheduler')]
    task_params['tsk_set']['optim']['lr_scheduler'][('step_size',10,'steps to decay learning rate')]
    task_params['tsk_set']['optim']['lr_scheduler'][('gamma',0.5,'factor to decay learning rate')]

    return task_params



def get_datapro_settings(datapro_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    datapro_params = pars.ParameterDict()

    if datapro_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        datapro_settings_filename = os.path.join(this_directory, r'../settings/data_settings.json')

    datapro_params.load_JSON( datapro_settings_filename )
    datapro_params[('datapro',{},'settings for the data process')]

    datapro_params['datapro'][('dataset', {}, 'general settings for dataset')]
    datapro_params['datapro']['dataset'][('dataset_name', 'lpba', 'name of the dataset: oasis2d, lpba, ibsr, cmuc')]
    datapro_params['datapro']['dataset'][('task_name', 'lpba_affined', 'task name for data process')]
    datapro_params['datapro']['dataset'][('data_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('label_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('output_path', '/playpen/zyshen/data/', "the path to save the processed data")]
    datapro_params['datapro']['dataset'][('prepare_data', False, 'prepare the data ')]
    datapro_params['datapro']['dataset'][('divided_ratio', (0.8, 0.1, 0.1), 'divided the dataset into train, val and test set by the divided_ratio')]
    datapro_params['datapro']['switch'][('switch_to_exist_task', False, 'switch to existed task without modify other datapro settings')]
    datapro_params['datapro']['switch'][('task_root_path', '/playpen/zyshen/data/oasis_inter_slicing90', 'path of existed processed data')]




    datapro_params['datapro'][('reg', {}, 'general settings for dataset')]
    datapro_params['datapro']['reg'][('sched', 'inter', "['inter'|'intra'], inter-personal or intra-personal")]
    datapro_params['datapro']['reg'][('all_comb', False, 'all possible pair combination ')]
    datapro_params['datapro']['reg'][('slicing', 100, 'the index to be sliced from the 3d image dataset, support lpba, ibsr, cmuc')]
    datapro_params['datapro']['reg'][('axis', 3, 'which axis needed to be sliced')]



    datapro_params['datapro'][('seg', {}, 'general settings for dataset')]
    datapro_params['datapro']['seg'][('num_crop_per_class_per_train_img',100, 'num_crop_per_class_per_train_img')]
    datapro_params['datapro']['seg'][('patch_size',[128, 128, 32], 'patch size')]
    datapro_params['datapro']['seg'][('sched', 'patched', "['patched'|'nopatched'], patched or whole image")]
    datapro_params['datapro']['seg'][('partition', {}, "settings for the partition")]
    datapro_params['datapro']['seg'][('transform', {}, 'settings for transform')]

    datapro_params['datapro']['seg']['partition'][('overlap_size',tuple([16,16,8]), 'overlap_size')]
    datapro_params['datapro']['seg']['partition'][('padding_mode', 'reflect', 'padding_mode')]
    datapro_params['datapro']['seg']['partition'][('mode', 'pred', 'eval or pred')]

    datapro_params['datapro']['seg']['transform']['transform_seq',[],"transform seqence list"]
    datapro_params['datapro']['seg']['transform'][('shared_info', {},'info shared by different transformers')]
    datapro_params['datapro']['seg']['transform'][('default', {}, 'only used when transform_seq is not given, get default transform setting')]
    datapro_params['datapro']['seg']['transform']['default'][('using_bspline_deform',False, 'using bspline transform')]
    datapro_params['datapro']['seg']['transform']['default'][('deform_target', 'padded','deform mode: global, local or padded')]
    datapro_params['datapro']['seg']['transform']['default'][('deform_scale', 1.0, 'deform scale')]
    datapro_params['datapro']['seg']['transform'][('bal_rand_crop', {}, 'settings for balanced random crop')]
    datapro_params['datapro']['seg']['transform'][('my_rand_crop', {}, 'settings for balanced random crop')]
    datapro_params['datapro']['seg']['transform']['my_rand_crop'][('scale_ratio', 0.05, 'scale_ratio for patch sampling')]
    datapro_params['datapro']['seg']['transform']['my_rand_crop'][('bg_label', 0, 'background label')]
    datapro_params['datapro']['seg']['transform']['my_rand_crop'][('crop_bg_ratio', 0.1, 'ratio of background crops')]
    datapro_params['datapro']['seg']['transform'][('my_bal_rand_crop', {}, 'settings for balanced random crop')]
    datapro_params['datapro']['seg']['transform']['my_bal_rand_crop'][('scale_ratio', 0.1, 'scale_ratio for patch sampling')]
    datapro_params['datapro']['seg']['transform'][('rand_rigid_trans', {}, 'settins for random_rigid_transform')]
    datapro_params['datapro']['seg']['transform']['rand_rigid_trans'][('transition',list([0.5]*3), 'transtion for each dimension')]
    datapro_params['datapro']['seg']['transform']['rand_rigid_trans'][('rotation',list([0.0]*3), 'rotation for each dimension')]
    datapro_params['datapro']['seg']['transform']['rand_rigid_trans'][('rigid_ratio',0.5, 'rigid ratio')]
    datapro_params['datapro']['seg']['transform']['rand_rigid_trans'][('rigid_mode','both', 'three mode: both , img, seg')]
    datapro_params['datapro']['seg']['transform']['my_bal_rand_crop'][('scale_ratio', 0.1, 'scale_ratio for patch sampling')]
    datapro_params['datapro']['seg']['transform'][('bspline_trans', {}, 'settins for bspline_transform')]
    datapro_params['datapro']['seg']['transform']['bspline_trans'][('bspline_order',3, 'bspline order')]
    datapro_params['datapro']['seg']['transform']['bspline_trans'][('deform_ratio',0.5, 'deform ratio')]
    datapro_params['datapro']['seg']['transform']['bspline_trans'][('deform_scale',1.0, 'deform scale')]
    datapro_params['datapro']['seg']['transform']['bspline_trans'][('interpolator',"BSpline", 'interpolation sched, linear or Bspline')]
    datapro_params['datapro']['seg']['transform'][('gaussian_blur', {}, 'settins for gaussian_blur')]
    datapro_params['datapro']['seg']['transform']['gaussian_blur'][('blur_ratio',1.0, 'blur ratio')]
    datapro_params['datapro']['seg']['transform']['gaussian_blur'][('gaussian_var',0.5, 'gaussian_var ')]
    datapro_params['datapro']['seg']['transform']['gaussian_blur'][('gaussian_width',1, 'gaussian_width')]
    datapro_params['datapro']['seg']['transform']['gaussian_blur'][('maximumError',0.9, 'maximumError')]
    datapro_params['datapro']['seg']['transform'][('bilateral_filter', {}, 'settins for bilateral_filter')]
    datapro_params['datapro']['seg']['transform']['bilateral_filter'][('bilateral_ratio',1.0, 'bilateral_ratio ratio')]
    datapro_params['datapro']['seg']['transform']['bilateral_filter'][('domain_sigma',0.2, 'domain_sigma')]
    datapro_params['datapro']['seg']['transform']['bilateral_filter'][('range_sigma',0.06, 'range_sigma')]


    return datapro_params



def get_respro_settings(respro_settings_filename = None):
    respro_params = pars.ParameterDict()

    if respro_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        respro_settings_filename = os.path.join(this_directory, r'../settings/respro_settings.json')

    respro_params.load_JSON(respro_settings_filename)
    respro_params[('respro',{},'settings for the results process')]

    respro_params['respro'][('expr_name', 'reg', 'name of experiment')]
    respro_params['respro'][('visualize', True, 'if set to true intermediate results are visualized')]
    respro_params['respro'][('visualize_step', 5, 'Number of iterations between visualization output')]

    respro_params['respro'][('save_fig', False, 'save visualized results')]
    respro_params['respro'][('save_fig_path', '../data/saved_results', 'path of saved figures')]
    respro_params['respro'][('save_excel', True, 'save results in excel')]

    return respro_params




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




