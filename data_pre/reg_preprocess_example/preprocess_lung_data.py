import sys
import os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../easyreg'))
import numpy as np
import glob
from multiprocessing import Pool
import os
from tools.image_rescale import resize_input_img_and_save_it_as_tmp
from tools.visual_tools import save_3D_img_from_numpy
from mermaid.data_wrapper import MyTensor
import mermaid.smoother_factory  as sf
import mermaid.module_parameters as pars
import mermaid.fileio as fileio
from functools import partial
from easyreg.reg_data_utils import write_list_into_txt, get_file_name,generate_pair_name

def set_path_env(s_list, t_list, ls_list, lt_list,intermid_path, output_path):
    raw_path_list = []
    inter_path_list = []
    out_path_list = []
    num_pair = len(s_list)
    for i in range(num_pair):
        s_inter_path =os.path.join(intermid_path,os.path.split(s_list[i])[-1].replace('.nrrd','.nii.gz'))
        t_inter_path =os.path.join(intermid_path,os.path.split(t_list[i])[-1].replace('.nrrd','.nii.gz'))
        ls_inter_path =os.path.join(intermid_path,os.path.split(ls_list[i])[-1].replace('.nrrd','.nii.gz'))
        lt_inter_path =os.path.join(intermid_path,os.path.split(lt_list[i])[-1].replace('.nrrd','.nii.gz'))
        s_out_path = os.path.join(output_path, os.path.split(s_list[i])[-1].replace('.nrrd','.nii.gz'))
        t_out_path = os.path.join(output_path, os.path.split(t_list[i])[-1].replace('.nrrd','.nii.gz'))
        ls_out_path = os.path.join(output_path, os.path.split(ls_list[i])[-1].replace('.nrrd','.nii.gz'))
        lt_out_path = os.path.join(output_path, os.path.split(lt_list[i])[-1].replace('.nrrd','.nii.gz'))
        raw_path_list.append(dict(source=s_list[i],lsource=ls_list[i],target=t_list[i],ltarget=lt_list[i]))
        inter_path_list.append(dict(source=s_inter_path,lsource=ls_inter_path,target=t_inter_path,ltarget=lt_inter_path))
        out_path_list.append(dict(source=s_out_path,lsource=ls_out_path,target=t_out_path,ltarget=lt_out_path))
    return raw_path_list, inter_path_list, out_path_list



def resize_and_save_img_pair(input,output,fixed_sz):
    resize_input_img_and_save_it_as_tmp(input['source'],fixed_sz=fixed_sz,is_label=False,keep_physical=True,saving_path=output['source'])
    resize_input_img_and_save_it_as_tmp(input['lsource'],fixed_sz=fixed_sz,is_label=True,keep_physical=True,saving_path=output['lsource'])
    resize_input_img_and_save_it_as_tmp(input['target'],fixed_sz=fixed_sz,is_label=False,keep_physical=True,saving_path=output['target'])
    resize_input_img_and_save_it_as_tmp(input['ltarget'],fixed_sz=fixed_sz,is_label=True,keep_physical=True,saving_path=output['ltarget'])

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

def get_pair_and_info(pth):
    source, info = file_io_read_img(pth['source'],is_label=False,normalize_spacing=True,normalize_intensities=True,squeeze_image=True,adaptive_padding=4)
    lsource, _ = file_io_read_img(pth['lsource'],is_label=True,normalize_spacing=True,normalize_intensities=False,squeeze_image=True,adaptive_padding=4)
    target, _ = file_io_read_img(pth['target'],is_label=False,normalize_spacing=True,normalize_intensities=True,squeeze_image=True,adaptive_padding=4)
    ltarget, _ = file_io_read_img(pth['ltarget'],is_label=True,normalize_spacing=True,normalize_intensities=False,squeeze_image=True,adaptive_padding=4)
    return source, lsource, target, ltarget, info


def file_io_read_img(path, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True, adaptive_padding=4):
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, normalized_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image,adaptive_padding)
    if normalize_spacing:
        spacing = normalized_spacing
    else:
        spacing = spacing
    info = { 'spacing':spacing, 'img_size': im.shape}
    return im, info

def subprocess(idxes,file_raw_path,inter_output_path,output_path,fixed_sz=None):
    for idx in idxes:
        resize_and_save_img_pair(file_raw_path[idx],inter_output_path[idx],fixed_sz=fixed_sz)
        source, lsource, target, ltarget, info = get_pair_and_info(inter_output_path[idx])
        source = np.clip(source,0,None)
        target = np.clip(target,0,None)
        spacing = info['spacing']
        spacing_flip = np.flip(spacing,axis=0)
        save_3D_img_from_numpy(source,output_path[idx]['source'],spacing_flip)
        save_3D_img_from_numpy(lsource,output_path[idx]['lsource'],spacing_flip)
        save_3D_img_from_numpy(target,output_path[idx]['target'],spacing_flip)
        save_3D_img_from_numpy(ltarget,output_path[idx]['ltarget'],spacing_flip)



all_exp_imgs = glob.glob('/playpen/zpd/lung_registration_example/UNCRegistration/**/*_EXP_STD_*_COPD.nrrd',recursive=True)
all_insp_imgs = glob.glob('/playpen/zpd/lung_registration_example/UNCRegistration/**/*_INSP_STD_*_COPD.nrrd',recursive=True)
all_lexp_imgs = glob.glob('/playpen/zpd/lung_registration_example/UNCRegistration/**/*_EXP_STD_*_COPD_wholeLungLabelMap.nrrd',recursive=True)
all_linsp_imgs = glob.glob('/playpen/zpd/lung_registration_example/UNCRegistration/**/*_INSP_STD_*_wholeLungLabelMap_cleaned.nii.gz',recursive=True)
inter_path ='/playpen/zyshen/lung_data/intermid'
output_path = '/playpen/zyshen/lung_data/resample_160'
output_txt_path  = '/playpen/zyshen/data/lung_160/'

header_exp_list =[]
header_insp_list = []
for i,exp in enumerate(all_exp_imgs):
    header_exp = os.path.split(exp)[-1].split('_')[0]
    header_exp_list.append(header_exp)
for i, insp in enumerate(all_insp_imgs):
    header_insp = os.path.split(all_insp_imgs[i])[-1].split('_')[0]
    header_insp_list.append(header_insp)

shared_set = set(header_insp_list).intersection(header_exp_list)
idx_to_remove=[]
for i in range(len(all_exp_imgs)):
    header_exp = os.path.split(all_exp_imgs[i])[-1].split('_')[0]
    if header_exp  not in shared_set:
        idx_to_remove.append(i)
for i in idx_to_remove:
    all_exp_imgs.remove(all_exp_imgs[i])
    all_lexp_imgs.remove(all_lexp_imgs[i])
idx_to_remove=[]
for i in range(len(all_insp_imgs)):
    header_insp = os.path.split(all_insp_imgs[i])[-1].split('_')[0]
    if header_insp not in shared_set:
        idx_to_remove.append(i)
for i in idx_to_remove:
    all_insp_imgs.remove(all_insp_imgs[i])
    all_linsp_imgs.remove(all_linsp_imgs[i])

for i in range(len(all_exp_imgs)):
    header_exp = os.path.split(all_exp_imgs[i])[-1].split('_')[0]
    header_insp = os.path.split(all_insp_imgs[i])[-1].split('_')[0]
    assert header_exp == header_insp



process_img = True
generate_txt = True
raw_path_list,inter_path_list, out_path_list= set_path_env(all_exp_imgs,all_insp_imgs,all_lexp_imgs,all_linsp_imgs,inter_path,output_path)

desired_sz = [160, 160, 160]
f = partial(subprocess,file_raw_path=raw_path_list,inter_output_path=inter_path_list,output_path=out_path_list,fixed_sz=desired_sz)
indexes = list(range(len(raw_path_list)))

number_of_workers = 8
boundary_list = []
config = dict()
index_partitions = np.array_split(indexes, number_of_workers)
if process_img:
    with Pool(processes=number_of_workers) as pool:
        pool.map(f, index_partitions)
if generate_txt:
    source_list = [f['source'] for f in out_path_list]
    target_list = [f['target'] for f in out_path_list]
    lsource_list = [f['lsource'] for f in out_path_list]
    ltarget_list = [f['ltarget'] for f in out_path_list]
    file_num = len(source_list)
    file_list = [[source_list[i], target_list[i], lsource_list[i], ltarget_list[i]] for i in
                     range(file_num)]
    os.makedirs(os.path.join(output_txt_path, 'reg/test'), exist_ok=True)
    os.makedirs(os.path.join(output_txt_path, 'reg/res'), exist_ok=True)
    pair_txt_path = os.path.join(output_txt_path, 'reg/test/pair_path_list.txt')
    fn_txt_path = os.path.join(output_txt_path, 'reg/test/pair_name_list.txt')
    fname_list = [generate_pair_name([file_list[i][0],file_list[i][1]]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path, file_list)
    write_list_into_txt(fn_txt_path, fname_list)