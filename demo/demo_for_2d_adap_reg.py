""""
demo on RDMM on 2d image registration
the function include:

get_pair_list
get_init_weight_list
get_input
get_mermaid_setting
do_registration
visualize_res

"""

import torch
import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.simple_interface as SI
from mermaid.pyreg.data_wrapper import AdaptVal
import numpy as np
from multiprocessing import *
import progressbar as pb
from functools import partial
from data_pre.reg_data_utils import read_txt_into_list
os.environ["CUDA_VISIBLE_DEVICES"] = ''

mermaid_setting_path = '../mermaid/demos/sample_generation/cur_settings_adpt_lddmm_for_synth.json'


def get_pair_list(folder_path):
    pair_path = os.path.join(folder_path,'pair_path_list.txt')
    fname_path = os.path.join(folder_path,'pair_name_list.txt')
    pair_path_list = read_txt_into_list(pair_path)
    pair_name_list = read_txt_into_list(fname_path)
    return pair_path_list, pair_name_list

def get_init_weight_list(folder_path):
    weight_path = os.path.join(folder_path,'pair_weight_path_list.txt')
    init_weight_path = read_txt_into_list(weight_path)
    return init_weight_path

def get_mermaid_setting(path,output_path):
    params = pars.ParameterDict()
    params.load_JSON(path)
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,'mermaid_setting.json')
    params.write_JSON(output_path,save_int=False)

def setting_intermid_saving(expr_folder,pair_name,expr_name=''):
    extra_info = {}
    extra_info['expr_name'] = expr_name
    extra_info['visualize']=False
    extra_info['save_fig']=True
    extra_info['save_fig_path']=os.path.join(expr_folder,'intermid')
    extra_info['save_fig_num'] = -1
    extra_info['save_excel'] =False
    extra_info['pair_name'] = [pair_name]
    extra_info['batch_id'] = [pair_name]
    return extra_info

def affine_optimization(moving,target,spacing,fname_list,l_moving=None,l_target=None):
    si = SI.RegisterImagePair()
    si.set_light_analysis_on(True)
    extra_info={}
    extra_info['pair_name'] = fname_list
    extra_info['batch_id'] = fname_list
    si.opt = None
    si.set_initial_map(None)
    si.register_images(moving, target, spacing,extra_info=extra_info,LSource=l_moving,LTarget=l_target,
                            model_name='affine_map',
                            map_low_res_factor=1.0,
                            nr_of_iterations=100,
                            visualize_step=None,
                            optimizer_name='sgd',
                            use_multi_scale=True,
                            rel_ftol=0,
                            similarity_measure_type='lncc',
                            similarity_measure_sigma=1.,
                            params ='../mermaid/demos/sample_generation/cur_settings_affine.json')
    output = si.get_warped_image()
    phi = si.opt.optimizer.ssOpt.get_map()
    disp = si.opt.optimizer.ssOpt.model.Ab
    # phi = phi*2-1
    phi = phi.detach().clone()
    return output.detach_(), phi.detach_(), disp.detach_(), si


def nonp_optimization(si, moving,target,spacing,fname,l_moving=None,l_target=None, init_weight= None,expr_folder= None):
    affine_map = None
    if si is not None:
        affine_map = si.opt.optimizer.ssOpt.get_map()

    si =  SI.RegisterImagePair()
    si.set_light_analysis_on(False)
    extra_info = setting_intermid_saving(expr_folder,fname)
    si.opt = None
    if affine_map is not None:
        si.set_initial_map(affine_map.detach())
    si.set_weight_map(init_weight.detach(),freeze_weight=True)

    si.register_images(moving, target, spacing, extra_info=extra_info, LSource=l_moving,
                            LTarget=l_target,
                            map_low_res_factor=0.5,
                            model_name='lddmm_adapt_smoother_map',#'lddmm_shooting_map',
                            nr_of_iterations=100,
                            visualize_step=5,
                            optimizer_name='lbfgs_ls',
                            use_multi_scale=True,
                            rel_ftol=0,
                            similarity_measure_type='lncc',
                            similarity_measure_sigma=1,
                            params=mermaid_setting_path)
    output = si.get_warped_image()
    phi = si.opt.optimizer.ssOpt.get_map()
    model_param = si.get_model_parameters()
    m, weight_map = model_param['m'], model_param['local_weights']

    return output.detach_(), phi.detach_(),m.detach(), weight_map.detach()


def get_input(img_pair,weight_pair):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    sw_path, tw_path = weight_pair
    moving_init_weight = torch.load(sw_path)
    target_init_weight = torch.load(tw_path)
    spacing = 1. / (np.array(moving.shape[2:]) - 1)
    return moving, target, spacing, moving_init_weight,target_init_weight

def warp_data(data_list):
    return [AdaptVal(data) for data in data_list]



def do_single_pair_registration(pair,pair_name, weight_pair, do_affine=True,expr_folder=None):
    moving, target, spacing, moving_init_weight, _ =get_input(pair,weight_pair)
    moving, target, moving_init_weight = warp_data([moving, target, moving_init_weight])
    si = None
    if do_affine:
        af_img, af_map, af_param, si =affine_optimization(moving,target,spacing,pair_name)
    return_val = nonp_optimization(si, moving, target, spacing, pair_name,init_weight=moving_init_weight,expr_folder=expr_folder)
    analysis_on_res(return_val, pair_name, expr_folder)

def do_pair_registration(pair_list, pair_name_list, weight_pair_list,do_affine=True,expr_folder=None):
    num_pair = len(pair_list)
    for i in range(num_pair):
        do_single_pair_registration(pair_list[i],pair_name_list[i],weight_pair_list[i],do_affine=do_affine,expr_folder=expr_folder)




def visualize_res(res, saving_path=None):
    pass



def analysis_on_res(res,pair_name, expr_folder):
    ana_path = os.path.join(expr_folder,'analysis')
    ana_path = os.path.join(ana_path,pair_name)
    os.makedirs(ana_path,exist_ok=True)
    output, phi, m, weight_map = res
    torch.save(phi.cpu(),os.path.join(ana_path,'phi.pt'))
    torch.save(m.cpu(),os.path.join(ana_path,'m.pt'))
    torch.save(weight_map.cpu(),os.path.join(ana_path,'weight_map.pt'))








def sub_process(index,pair_list, pair_name_list, weight_pair_list,do_affine,expr_folder):
    num_pair = len(index)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_pair).start()
    count = 0
    for i in index:
        do_single_pair_registration(pair_list[i],pair_name_list[i],weight_pair_list[i],do_affine=do_affine,expr_folder=expr_folder)
        count += 1
        pbar.update(count)
    pbar.finish()




def demo():
    data_folder = '/playpen/zyshen/data/syn_data/test'
    expr_folder = '/playpen/zyshen/data/syn_data/expr6/res'
    do_affine = False
    os.makedirs(expr_folder,exist_ok=True)
    pair_path_list, pair_name_list = get_pair_list(data_folder)
    init_weight_path_list = get_init_weight_list(data_folder)
    get_mermaid_setting(mermaid_setting_path,expr_folder)


    num_of_workers = 8 #for unknown reason, multi-thread not work
    num_files = len(pair_name_list)
    if num_of_workers > 1:
        sub_p = partial(sub_process, pair_list=pair_path_list, pair_name_list=pair_name_list,
                        weight_pair_list=init_weight_path_list, do_affine=do_affine, expr_folder=expr_folder)
        split_index = np.array_split(np.array(range(num_files)), num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=sub_p, args=(split_index[i],))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
    else:
        do_pair_registration(pair_path_list, pair_name_list, init_weight_path_list,do_affine=do_affine,expr_folder=expr_folder)

demo()