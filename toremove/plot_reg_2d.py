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
import matplotlib as matplt
matplt.use('Agg')
import SimpleITK as sitk

import torch
import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
torch.cuda.set_device(1)
import mermaid.module_parameters as pars
import mermaid.simple_interface as SI
from mermaid.model_evaluation import evaluate_model, evaluate_model_low_level_interface
from mermaid.data_wrapper import AdaptVal,MyTensor
from mermaid.metrics import get_multi_metric
import mermaid.utils as utils
import tools.visual_tools as vt
from skimage.draw._random_shapes import _generate_random_colors
import mermaid.finite_differences as fdt
import numpy as np
from multiprocessing import *
import progressbar as pb
from functools import partial
from data_pre.reg_data_utils import read_txt_into_list
os.environ["CUDA_VISIBLE_DEVICES"] = ''
model_name = 'lddmm_adapt_smoother_map'#'lddmm_adapt_smoother_map'#'lddmm_shooting_map' #svf_vector_momentum_map


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



def modify_setting(params,stds,weights):
    if stds is not None:
        params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_stds']=stds
    if weights is not None:
        params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_weights']=weights
    return params

def get_mermaid_setting(path,output_path, modify=False,stds=None,weights= None):
    params = pars.ParameterDict()
    params.load_JSON(path)
    if modify:
        params = modify_setting(params,stds,weights)
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,'mermaid_setting.json')
    params.write_JSON(output_path,save_int=False)



def setting_visual_saving(expr_folder,pair_name,expr_name='',folder_name='intermid'):
    extra_info = {}
    extra_info['expr_name'] = expr_name
    extra_info['visualize']=True
    extra_info['save_fig']=True
    extra_info['save_fig_path']=os.path.join(expr_folder,folder_name)
    extra_info['save_fig_num'] = -1  #todo here should be -1
    extra_info['save_excel'] =False
    extra_info['pair_name'] = [pair_name] if not isinstance(pair_name,list) else pair_name
    extra_info['batch_id'] = [pair_name] if not isinstance(pair_name,list) else pair_name
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


def nonp_optimization(si, moving,target,spacing,fname,l_moving=None,l_target=None, init_weight= None,expr_folder= None,mermaid_setting_path=None):
    affine_map = None
    if si is not None:
        affine_map = si.opt.optimizer.ssOpt.get_map()

    si =  SI.RegisterImagePair()
    si.set_light_analysis_on(True)  # todo here should be False
    extra_info=None
    #extra_info = setting_visual_saving(expr_folder,fname)
    si.opt = None
    if affine_map is not None:
        si.set_initial_map(affine_map.detach())
    if init_weight is not None:
        si.set_weight_map(init_weight.detach(),freeze_weight=True)

    si.register_images(moving, target, spacing, extra_info=extra_info, LSource=l_moving,
                            LTarget=l_target,
                            map_low_res_factor=0.5,
                            model_name=model_name,#'lddmm_shooting_map',
                            nr_of_iterations=100,
                            visualize_step=None , # todo here should be 30
                            optimizer_name='sgd',
                            use_multi_scale=True,
                            rel_ftol=0,
                            similarity_measure_type='lncc',
                            similarity_measure_sigma=1,
                            params=mermaid_setting_path)
    output = si.get_warped_image()
    phi = si.opt.optimizer.ssOpt.get_map()
    model_param = si.get_model_parameters()
    if model_name =='lddmm_adapt_smoother_map':
        m, weight_map = model_param['m'], model_param['local_weights']
        return output.detach_(), phi.detach_(), m.detach(), weight_map.detach()
    else:
        m = model_param['m']
        return output.detach_(), phi.detach_(), m.detach(), None


def compute_jacobi(phi,spacing):
    spacing = spacing  # the disp coorindate is [-1,1]
    fd = fdt.FD_torch(spacing)
    dfx = fd.dXc(phi[:, 0, ...])
    dfy = fd.dYc(phi[:, 1, ...])
    jacobi_det = dfx * dfy
    jacobi_det = jacobi_det.cpu().detach().numpy()
    crop_range = 5
    jacobi_det_croped = jacobi_det[:, crop_range:-crop_range, crop_range:-crop_range]
    jacobi_abs = - np.sum(jacobi_det_croped[jacobi_det_croped < 0.])  #
    print("the current  average jacobi is {}".format(jacobi_abs))
    return jacobi_abs




def get_batch_input(img_pair_list, weight_pair_list):
    moving_list =[]
    target_list= []
    spacing=None
    moving_init_weight_list = [] if weight_pair_list else None
    target_init_weight_list = [] if weight_pair_list else None
    for i, img_pair in enumerate(img_pair_list):
        moving, target, spacing, moving_init_weight, target_init_weight \
            =get_input(img_pair,weight_pair=weight_pair_list[i] if weight_pair_list is not None else None)
        moving_list.append(moving)
        target_list.append(target)
        if weight_pair_list is not None:
            moving_init_weight_list.append(moving_init_weight)
            target_init_weight_list.append(target_init_weight)
    moving = torch.cat(moving_list,0)
    target = torch.cat(target_list,0)
    if weight_pair_list is not None:
        moving_init_weight = torch.cat(moving_init_weight_list,0)
        target_init_weight = torch.cat(target_init_weight_list,0)
    else:
        moving_init_weight = None
        target_init_weight = None
    return moving, target, spacing, moving_init_weight, target_init_weight


def get_input(img_pair,weight_pair=None):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    moving_init_weight= None
    target_init_weight= None
    if weight_pair is not None:
        sw_path, tw_path = weight_pair
        moving_init_weight = torch.load(sw_path)
        target_init_weight = torch.load(tw_path)
    spacing = 1. / (np.array(moving.shape[2:]) - 1)
    return moving, target, spacing, moving_init_weight,target_init_weight


def get_batch_analysis_input(img_pair_list, expr_folder,pair_name_list,color_image=False):
    moving_list = []
    target_list = []
    m_list = []
    phi_list = []
    weight_map_list = None
    if model_name == 'lddmm_adapt_smoother_map':
        weight_map_list = []
    for i, img_pair in enumerate(img_pair_list):
        moving, target, spacing, weight_map, phi, m \
            = get_analysis_input(img_pair_list[i],expr_folder,pair_name_list[i],color_image=color_image)
        moving_list.append(moving)
        target_list.append(target)
        phi_list.append(phi)
        m_list.append(m)
        if model_name == 'lddmm_adapt_smoother_map':
            weight_map_list.append(weight_map)
    moving = torch.cat(moving_list, 0)
    target = torch.cat(target_list, 0)
    phi = torch.cat(phi_list, 0)
    m = torch.cat(m_list, 0)
    weight_map =None
    if model_name == 'lddmm_adapt_smoother_map':
        weight_map = torch.cat(weight_map_list,0)
    return moving, target, spacing, weight_map, phi, m




def get_analysis_input(img_pair,expr_folder,pair_name,color_image=False):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    spacing = 1. / (np.array(moving.shape[2:]) - 1)
    ana_path = os.path.join(expr_folder, 'analysis')
    ana_path = os.path.join(ana_path, pair_name)
    phi = torch.load(os.path.join(ana_path, 'phi.pt'))
    m = torch.load(os.path.join(ana_path, 'm.pt'))
    if color_image:
        moving,color_info = generate_color_image(moving,random_seed=2016)
        target,color_info = generate_color_image(target, color_info=color_info)
        moving = MyTensor(moving)
        target = MyTensor(target)

    if model_name == 'lddmm_adapt_smoother_map':
        weight_map = torch.load(os.path.join(ana_path, 'weight_map.pt'))
    else:
        weight_map = None
    return moving, target, spacing, weight_map, phi, m





def generate_color_image(refer_image,num_channels=3,intensity_range=None,random_seed=None,color_info= None):

    if type(refer_image) is torch.Tensor:
        refer_image = refer_image.cpu().numpy()
        refer_image = np.squeeze(refer_image)
    image_shape = refer_image.shape
    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.ones(image_shape, dtype=np.uint8) * 0
    filled = np.zeros(image_shape, dtype=bool)
    refer_intensity = np.unique(refer_image)
    num_shapes = len(refer_intensity)
    if color_info is None:
        colors = _generate_random_colors(num_shapes, num_channels,
                                         intensity_range, random)
    else:
        colors = color_info
    for shape_idx in range(num_shapes):
        indices = np.where(refer_image==refer_intensity[shape_idx])

        if not filled[indices].any():
            image[indices] = colors[shape_idx]
            filled[indices] = True
            continue

    image = image.astype(np.float32)/intensity_range[0][1]
    image = np.transpose(image,[2,0,1])
    image_shape = [1]+ list(image.shape)
    image = image.reshape(*image_shape)
    return image, colors


def save_color_image(image,pair_name,target):
    pass



def warp_data(data_list):
    return [AdaptVal(data) if data is not None else None for data in data_list]



def do_single_pair_registration(pair,pair_name, weight_pair, do_affine=True,expr_folder=None,mermaid_setting_path=None):
    moving, target, spacing, moving_init_weight, _ =get_input(pair,weight_pair)
    moving, target, moving_init_weight = warp_data([moving, target, moving_init_weight])
    si = None
    if do_affine:
        af_img, af_map, af_param, si =affine_optimization(moving,target,spacing,pair_name)
    return_val = nonp_optimization(si, moving, target, spacing, pair_name,init_weight=moving_init_weight,expr_folder=expr_folder,mermaid_setting_path=mermaid_setting_path)
    save_single_res(return_val, pair_name, expr_folder)

def do_pair_registration(pair_list, pair_name_list, weight_pair_list,do_affine=True,expr_folder=None,mermaid_setting_path=None):
    moving, target, spacing, moving_init_weight, _ = get_batch_input(pair_list, weight_pair_list)
    moving, target, moving_init_weight = warp_data([moving, target, moving_init_weight])
    si = None
    if do_affine:
        af_img, af_map, af_param, si = affine_optimization(moving, target, spacing, pair_name_list)
    return_val = nonp_optimization(si, moving, target, spacing, pair_name_list, init_weight=moving_init_weight,
                                   expr_folder=expr_folder, mermaid_setting_path=mermaid_setting_path)
    save_batch_res(return_val, pair_name_list, expr_folder)



def visualize_res(res, saving_path=None):
    pass



def save_batch_res(res,pair_name_list, expr_folder):
    num_pair = len(pair_name_list)
    for i in range(num_pair):
        save_single_res([iterm[i:i+1,:] if iterm is not None else None for iterm in res], pair_name_list[i],expr_folder)

def save_single_res(res,pair_name, expr_folder):
    ana_path = os.path.join(expr_folder,'analysis')
    ana_path = os.path.join(ana_path,pair_name)
    os.makedirs(ana_path,exist_ok=True)
    output, phi, m, weight_map = res
    torch.save(phi.cpu(),os.path.join(ana_path,'phi.pt'))
    torch.save(m.cpu(),os.path.join(ana_path,'m.pt'))
    if weight_map is not None:
        torch.save(weight_map.cpu(),os.path.join(ana_path,'weight_map.pt'))








def sub_process(index,pair_list, pair_name_list, weight_pair_list,do_affine,expr_folder, mermaid_setting_path):
    num_pair = len(index)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_pair).start()
    count = 0
    for i in index:
        do_single_pair_registration(pair_list[i],pair_name_list[i],weight_pair_list[i] if weight_pair_list else None,do_affine=do_affine,expr_folder=expr_folder, mermaid_setting_path=mermaid_setting_path)
        count += 1
        pbar.update(count)
    pbar.finish()


def get_batch_label_img(img_pair_list):
    lmoving_list = []
    ltarget_list = []
    for img_pair in img_pair_list:
        lmoving, ltarget = get_labeled_image(img_pair)
        lmoving_list.append(lmoving)
        ltarget_list.append(ltarget)

    lmoving = torch.cat(lmoving_list,0)
    ltarget = torch.cat(ltarget_list,0)
    return lmoving, ltarget


def get_labeled_image(img_pair):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    moving_np  = moving.cpu().numpy()
    target_np = target.cpu().numpy()
    ind_value_list = np.unique(moving_np)
    ind_value_list_target = np.unique(target_np)
    assert len(set(ind_value_list)-set(ind_value_list_target))==0
    lmoving = torch.zeros_like(moving)
    ltarget = torch.zeros_like(target)
    ind_value_list.sort()
    for i,value in enumerate(ind_value_list):
        lmoving[moving==value]= i
        ltarget[target==value]= i

    return AdaptVal(lmoving),AdaptVal(ltarget)





def analyze_on_pair_res(pair_list,pair_name_list,expr_folder=None,color_image=False):
    moving, target, spacing, moving_init_weight, phi, m = get_batch_analysis_input(pair_list, expr_folder, pair_name_list,
                                                                             color_image=color_image)
    lmoving, ltarget = get_batch_label_img(pair_list)
    params = pars.ParameterDict()
    params.load_JSON(os.path.join(expr_folder, 'mermaid_setting.json'))
    individual_parameters = dict(m=m, local_weights=moving_init_weight)
    sz = np.array(moving.shape)
    saving_folder_root = os.path.join(expr_folder, 'analysis')
    saving_folder_list = []
    for pair_name  in pair_name_list:
        saving_folder = os.path.join(saving_folder_root, pair_name)
        saving_folder = os.path.join(saving_folder, 'res_analysis')
        saving_folder_list.append(saving_folder)
        os.makedirs(saving_folder, exist_ok=True)

    extra_info = {'fname': pair_name_list, 'saving_folder': saving_folder_list}
    visual_param = setting_visual_saving(expr_folder, pair_name_list, folder_name='color')
    # visual_param = None
    # extra_info = None
    res=  evaluate_model(moving, target, sz, spacing,
                       model_name=model_name,
                       use_map=True,
                       compute_inverse_map=False,
                       map_low_res_factor=0.5,
                       compute_similarity_measure_at_low_res=False,
                       spline_order=1,
                       individual_parameters=individual_parameters,
                       shared_parameters=None, params=params, extra_info=extra_info, visualize=False,
                       visual_param=visual_param, given_weight=True)
    phi = res[1]
    phi = AdaptVal(phi)
    lres = utils.compute_warped_image_multiNC(lmoving, phi, spacing, 0, zero_boundary=True)
    scores = get_multi_metric(lres, ltarget, rm_bg=True)['label_avg_res']['dice']
    jacobi_mean = compute_jacobi(phi, spacing)
    print("the average score of the registration is {}".format(scores.mean()))
    print("the average jacobi of the registration is {}".format(jacobi_mean))
    saving_folder = os.path.join(expr_folder, 'score')
    os.makedirs(saving_folder, exist_ok=True)
    np.save(os.path.join(saving_folder,'score.npy'),scores)
    saving_folder = os.path.join(expr_folder, 'jacobi')
    os.makedirs(saving_folder, exist_ok=True)
    np.save(os.path.join(saving_folder,'jacobi.npy'),jacobi_mean)
    with open(os.path.join(expr_folder,'res.txt'),'w+') as f:
        f.write("score {} ".format(scores.mean()))
        f.write("jacobi {} \n".format((jacobi_mean)))

    return scores.mean(), jacobi_mean






def demo_multi():
    import argparse

    parser = argparse.ArgumentParser(description='Registeration demo (include train and test)')
    parser.add_argument('--llf', required=False, type=bool, default=False,
                        help='run on long leaf')
    parser.add_argument('--expr_name', required=False, default='non_name',
                        help='the name of the experiment')
    parser.add_argument('--mermaid_setting_path', required=False, default='non_path',
                        help='the setting of mermaid')
    args = parser.parse_args()
    llf = args.llf
    if not llf:
        root_path = '/playpen/zyshen/data/syn_data'
    else:
        root_path = '/pine/scr/z/y/zyshen/expri/syn_data'

    expr_name = 'non_name_tv_weighted_no_omt3_1_multi_sm002/0_0'  #args.expr_name + '_multi'
    data_folder = os.path.join(root_path, 'test')
    expr_folder = os.path.join(root_path, expr_name)
    os.makedirs(expr_folder, exist_ok=True)
    pair_path_list, pair_name_list = get_pair_list(data_folder)
    pair_path_list = pair_path_list[1:10]
    pair_name_list = pair_name_list[1:10]


    do_grid_evaluation = False
    do_evaluation = True
    color_image = True
    num_stds =15
    num_weights =10
    score_matrix = np.zeros((num_stds,num_weights))
    jacobi_matrix = np.zeros((num_stds,num_weights))
    if model_name == 'lddmm_adapt_smoother_map':
        score_matrix = np.zeros((num_stds, 1))
        jacobi_matrix = np.zeros((num_stds, 1))

    expr_folder_root = expr_folder
    if do_grid_evaluation:
        for i in range(num_stds):
            for j in range(num_weights):
                expr_folder = os.path.join(expr_folder_root,'{}_{}'.format(i,j))
                os.makedirs(expr_folder, exist_ok=True)

                if do_evaluation:
                    score, jacobi = analyze_on_pair_res(pair_path_list, pair_name_list, expr_folder=expr_folder, color_image=color_image)
                    score_matrix[i,j]=score
                    jacobi_matrix[i,j]=jacobi
                    print("this is the {}, {} epoch".format(i,j))
                if model_name == 'lddmm_adapt_smoother_map':
                    break

        np.save(os.path.join(expr_folder_root,'score.npy'),score_matrix)
        np.save(os.path.join(expr_folder_root,'jacobi.npy'),jacobi_matrix)
    else:
        score, jacobi = analyze_on_pair_res(pair_path_list, pair_name_list, expr_folder=expr_folder,
                                            color_image=color_image)


demo_multi()
# python demo_for_2d_adap_reg_batch.py --expr_name="lddmm"