import numpy as np
from mermaid.data_wrapper import MyTensor, AdaptVal
import mermaid.smoother_factory  as sf
import mermaid.module_parameters as pars
import mermaid.fileio as fileio
import mermaid.multiscale_optimizer as MO

from tools.image_rescale import resize_input_img_and_save_it_as_tmp

def file_io_read_img(path, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True, adaptive_padding=4):
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image,adaptive_padding)
    if normalize_spacing:
        spacing = normalized_spacing
    else:
        spacing = spacing
    info = { 'spacing':spacing, 'img_size': im.shape}
    return im, info


def resize_and_save_img_pair(input,output,resize_factor):
    resize_input_img_and_save_it_as_tmp(input['source'],resize_factor=resize_factor,is_label=False,keep_physical=True,saving_path=output['source'])
    resize_input_img_and_save_it_as_tmp(input['lsource'],resize_factor=resize_factor,is_label=True,keep_physical=True,saving_path=output['lsource'])
    resize_input_img_and_save_it_as_tmp(input['target'],resize_factor=resize_factor,is_label=False,keep_physical=True,saving_path=output['target'])
    resize_input_img_and_save_it_as_tmp(input['ltarget'],resize_factor=resize_factor,is_label=True,keep_physical=True,saving_path=output['ltarget'])


def get_pair_and_info(pth):
    source, info = file_io_read_img(pth['source'],is_label=False,normalize_spacing=True,normalize_intensities=True,squeeze_image=True,adaptive_padding=4)
    lsource, _ = file_io_read_img(pth['lsource'],is_label=True,normalize_spacing=True,normalize_intensities=False,squeeze_image=True,adaptive_padding=4)
    target, _ = file_io_read_img(pth['target'],is_label=False,normalize_spacing=True,normalize_intensities=True,squeeze_image=True,adaptive_padding=4)
    ltarget, _ = file_io_read_img(pth['ltarget'],is_label=True,normalize_spacing=True,normalize_intensities=False,squeeze_image=True,adaptive_padding=4)
    return source, lsource, target, ltarget, info

def inverse_intensity(img):
    img = 1. - img
    return img

def get_single_gaussian_smoother(gaussian_std,sz,spacing):
    s_m_params = pars.ParameterDict()
    s_m_params['smoother']['type'] = 'gaussian'
    s_m_params['smoother']['gaussian_std'] = gaussian_std
    s_m = sf.SmootherFactory(sz, spacing).create_smoother(s_m_params)
    return s_m


def get_source_init_weight_map(source,spacing, lsource,default_multi_gaussian_weights,multi_gaussian_weights):
    sz = source.shape
    nr_of_mg_weights = len(default_multi_gaussian_weights)
    sh_weights = [1] + [nr_of_mg_weights] + list(sz)
    weights = np.zeros(sh_weights, dtype='float32')
    for g in range(nr_of_mg_weights):
        weights[:, g, ...] = default_multi_gaussian_weights[g]
    indexes = np.where(lsource>0)
    for g in range(nr_of_mg_weights):
        weights[:, g, indexes[0], indexes[1],indexes[2]] = multi_gaussian_weights[g]
    weights = MyTensor(weights)
    local_smoother  = get_single_gaussian_smoother(0.02,sz,spacing)
    sm_weight = local_smoother.smooth(weights)
    return sm_weight


def load_settings(setting_file):
    params = pars.ParameterDict()
    params.load_JSON(setting_file)
    return params


def do_registration(I0,I1,params,spacing,weight=None,multi_scale=True):
    sz = np.array(I0.shape)
    if not multi_scale:
        so = MO.SimpleSingleScaleRegistration(I0, I1, spacing, sz, params)
    else:
        so = MO.SimpleMultiScaleRegistration(I0, I1, spacing, sz, params)
    so.get_optimizer().set_visualization(True)
    so.get_optimizer().set_visualize_step(10)
    so.set_light_analysis_on(True)
    so.optimizer.set_model(params['model']['registration_model']['type'])
    so.optimizer.set_initial_weight_map(weight.cuda(),freeze_weight=True)
    so.register()

file_raw_path =dict(source='/playpen/zpd/lung_registration_example/UNCRegistration/12593R/12593R_EXP_STD_NJC_COPD.nrrd',
                lsource='/playpen/zpd/lung_registration_example/UNCRegistration/12593R/12593R_EXP_STD_NJC_COPD_wholeLungLabelMap.nrrd',
                target='/playpen/zpd/lung_registration_example/UNCRegistration/12593R/12593R_INSP_STD_NJC_COPD.nrrd',
                ltarget='/playpen/zpd/lung_registration_example/UNCRegistration/12593R/12593R_INSP_STD_NJC_COPD_wholeLungLabelMap_cleaned.nii.gz')

file_path = dict(source='/playpen/zyshen/debugs/lung_reg/12593R_EXP_STD_NJC_COPD_img.nii.gz',
                lsource='/playpen/zyshen/debugs/lung_reg/12593R_EXP_STD_NJC_COPD_label.nii.gz',
                target='/playpen/zyshen/debugs/lung_reg/12593R_INSP_STD_NJC_COPD_img.nii.gz',
                ltarget='/playpen/zyshen/debugs/lung_reg/12593R_INSP_STD_NJC_COPD_label.nii.gz')

file_path_inversed = dict(source='/playpen/zyshen/debugs/lung_reg/12593R_EXP_STD_NJC_COPD_nom_img.nii.gz',
                lsource='/playpen/zyshen/debugs/lung_reg/12593R_EXP_STD_NJC_COPD_nom_label.nii.gz',
                target='/playpen/zyshen/debugs/lung_reg/12593R_INSP_STD_NJC_COPD_nom_img.nii.gz',
                ltarget='/playpen/zyshen/debugs/lung_reg/12593R_INSP_STD_NJC_COPD_nom_label.nii.gz')

setting_file = '/playpen/zyshen/reg_clean/mermaid_settings/cur_settings_adpt_lddmm_for_synth.json'



desired_sz = [256,256,256]
resize_factor = [1./4,1./4,1./4]
resize_and_save_img_pair(file_raw_path,file_path,resize_factor=resize_factor)
source, lsource, target, ltarget, info = get_pair_and_info(file_path)
source = np.clip(source,0,None)
target = np.clip(target,0,None)
spacing = info['spacing']
spacing_flip = np.flip(spacing,axis=0)
# save_3D_img_from_numpy(source,file_path_inversed['source'],spacing_flip)
# save_3D_img_from_numpy(lsource,file_path_inversed['lsource'],spacing_flip)
# save_3D_img_from_numpy(target,file_path_inversed['target'],spacing_flip)
# save_3D_img_from_numpy(ltarget,file_path_inversed['ltarget'],spacing_flip)

multi_gaussian_weights = np.array([0.2,0.8,0.0,0.0])
default_multi_gaussian_weights = np.array([0,0,0,1.])
source_weight_map=get_source_init_weight_map(source,spacing, lsource,default_multi_gaussian_weights,multi_gaussian_weights)
I0 = MyTensor(source).view([1,1]+list(source.shape))
I1 = MyTensor(target).view([1,1]+list(target.shape))
params =load_settings(setting_file)
do_registration(I0,I1,params,spacing,weight=source_weight_map,multi_scale=True)
