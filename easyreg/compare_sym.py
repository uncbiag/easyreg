import torch
import numpy as np
import sys,os
import SimpleITK as sitk

from easyreg.net_utils import Bilinear

import ants
from .nifty_reg_utils import nifty_reg_resample
import subprocess
import nibabel as nib
from mermaid.utils import identity_map_multiN

# record_path ='/playpen/zyshen/debugs/compare_sym'
# moving_img_path = os.path.join('/playpen/zyshen/debugs/compare_sym', 'source.nii.gz')
# if not os.path.exists(record_path):
#     os.mkdir(record_path)
# dim = 3
# szEx =np.array([80,192,192]) # size of the desired images: (sz)^dim
# I0,spacing = eg.CreateGrid(dim,add_noise_to_bg=False).create_image_single(szEx, None)  # create a default image size with two sample squares
# sz = np.array(I0.shape)
# I0 = np.squeeze(I0)
# sitk.WriteImage(sitk.GetImageFromArray(I0),moving_img_path)
#





def init_source_image(record_path):
    img_size = [80,192,192]
    spacing = 1. / (np.array(img_size) - 1)
    identity_map = identity_map_multiN([1,1]+img_size, spacing)
    print(identity_map.shape)
    id_path  =  os.path.join(record_path, 'identity.nii.gz')
    id_x_pth = id_path.replace('identity','identity_x')
    id_y_pth = id_path.replace('identity', 'identity_y')
    id_z_pth = id_path.replace('identity','identity_z')

    sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,0]),id_x_pth)
    sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,1]),id_y_pth)
    sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,2]),id_z_pth)
    return [id_x_pth, id_y_pth,id_z_pth]


def __inverse_name(name):
    """get the name of the inversed registration pair"""
    name = name + '_inverse'
    return name

def compute_sym_metric(refer, output, shrink_margin=(10,20,20)):
    return np.mean((refer[shrink_margin[0]:-shrink_margin[0],shrink_margin[1]:-shrink_margin[1],shrink_margin[2]:-shrink_margin[2]]
                    -output[shrink_margin[0]:-shrink_margin[0],shrink_margin[1]:-shrink_margin[1],shrink_margin[2]:-shrink_margin[2]])**2)

def sitk_grid_sampling(fixed,moving, displacement,is_label=False):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(displacement)
    out = resampler.Execute(moving)
    return out

def cal_ants_sym(record_path,fname,moving_img_path):
    inv_fname = __inverse_name(fname)
    disp_pth = [os.path.join(record_path, fname + '_disp.nii.gz'),os.path.join(record_path, fname + '_affine.mat')]
    inv_disp_pth = [os.path.join(record_path, inv_fname + '_disp.nii.gz'),os.path.join(record_path, inv_fname + '_affine.mat')]
    # source1 = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))
    # source2 = ants.image_read(moving_img_path).numpy()
    # source2 = np.transpose(source2)
    source = ants.image_read(moving_img_path)
    target = ants.image_read(moving_img_path)
    output = ants.apply_transforms(fixed=target, moving=source, transformlist=disp_pth)
    output = ants.apply_transforms(fixed=target, moving=output, transformlist=inv_disp_pth)
    output = output.numpy()
    output = np.transpose(output)
    return output


def cal_demons_sym(record_path,fname,moving_img_path):


    inv_fname = __inverse_name(fname)
    disp_pth = os.path.join(record_path, fname + '_disp.nii.gz')
    inv_disp_pth = os.path.join(record_path, inv_fname + '_disp.nii.gz')
    disp = sitk.ReadImage(disp_pth)
    inv_disp = sitk.ReadImage(inv_disp_pth)
    tx = sitk.DisplacementFieldTransform(disp)
    inv_tx = sitk.DisplacementFieldTransform(inv_disp)
    target =sitk.ReadImage(moving_img_path)



    af_txt = os.path.join(record_path,fname+'_af.txt')
    af_pth = os.path.join(record_path,'af_output.nii.gz')
    cmd = nifty_reg_resample(ref=moving_img_path, flo=moving_img_path, trans=af_txt, res=af_pth, inter=0)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()



    source = sitk.ReadImage(af_pth)
    output = sitk_grid_sampling(target, source, tx, is_label=False)
    forward_output = os.path.join(record_path,'forward_output.nii.gz')
    sitk.WriteImage(output,forward_output)


    inv_af_txt = os.path.join(record_path, inv_fname + '_af.txt')
    for_inv_af_pth = os.path.join(record_path,'for_inv_af_output.nii.gz')
    cmd = nifty_reg_resample(ref=moving_img_path, flo=forward_output, trans=inv_af_txt, res=for_inv_af_pth, inter=0)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    source = sitk.ReadImage(for_inv_af_pth)
    output = sitk_grid_sampling(target, source, inv_tx, is_label=False)
    output = sitk.GetArrayFromImage(output)
    return output


def cal_demons_sym_new(record_path,fname,moving_img_path):


    inv_fname = __inverse_name(fname)
    disp_pth = os.path.join(record_path, fname + '_disp.nii.gz')
    inv_disp_pth = os.path.join(record_path, inv_fname + '_disp.nii.gz')
    disp = sitk.ReadImage(disp_pth)
    inv_disp = sitk.ReadImage(inv_disp_pth)
    tx = sitk.DisplacementFieldTransform(disp)
    inv_tx = sitk.DisplacementFieldTransform(inv_disp)
    target =sitk.ReadImage(moving_img_path)

    source = sitk.ReadImage(moving_img_path)
    output = sitk_grid_sampling(target, source, tx, is_label=False)
    output = sitk_grid_sampling(target, output, inv_tx, is_label=False)
    output = sitk.GetArrayFromImage(output)
    return output





def cal_nifty_sym(record_path,fname,moving_img_path):
    inv_fname = __inverse_name(fname)
    def_pth = os.path.join(record_path, fname + '_deformation.nii.gz')
    inv_def_pth = os.path.join(record_path, inv_fname + '_deformation.nii.gz')
    output_path = os.path.join(record_path, 'output_sym.nii.gz')

    cmd = '' + nifty_reg_resample(ref=moving_img_path, flo=moving_img_path, trans=def_pth, res=output_path, inter=0)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    cmd = '' + nifty_reg_resample(ref=moving_img_path, flo=output_path, trans=inv_def_pth, res=output_path, inter=0)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    output = sitk.ReadImage(output_path)
    output = sitk.GetArrayFromImage(output)
    return output

def cal_mermaid_sym(record_path,fname, moving_img_path):
    inv_fname = __inverse_name(fname)
    def_pth = os.path.join(record_path, fname + '_phi.nii.gz')
    inv_def_pth = os.path.join(record_path, inv_fname + '_phi.nii.gz')
    source = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))
    source=np.expand_dims(np.expand_dims(source, 0), 0)
    source = torch.Tensor(source)
    phi = nib.load(def_pth).get_fdata()
    inv_phi = nib.load(inv_def_pth).get_fdata()
    phi = torch.Tensor(np.expand_dims(phi,0))
    inv_phi = torch.Tensor(np.expand_dims(inv_phi,0))
    b1 = Bilinear(zero_boundary=True,using_scale=False)
    b2 = Bilinear(zero_boundary=True,using_scale=False)
    output  = b1(source, phi)
    output = b2(output,inv_phi)
    return np.squeeze(output.detach().numpy())





def cal_sym(opt,dataloaders,task_name=''):
    record_path = opt['tsk_set']['path']['record_path']
    model_name = opt['tsk_set']['model']
    orginal_img_path_list = init_source_image(record_path)
        #os.path.join('/playpen/zyshen/debugs/compare_sym', 'source.nii.gz')

    phases=['test']
    orginal_list = [sitk.GetArrayFromImage(sitk.ReadImage(pth)) for pth in orginal_img_path_list]
    cal_sym_func = None
    if model_name == 'ants':
        cal_sym_func = cal_ants_sym
    if model_name == 'demons':
        cal_sym_func = cal_demons_sym_new
    if model_name =='nifty_reg':
        cal_sym_func = cal_nifty_sym
    if model_name =='reg_net' or model_name == 'mermaid_iter':
        cal_sym_func = cal_mermaid_sym



    for phase in phases:
        num_samples = len(dataloaders[phase])
        records_sym_val_np = np.zeros(num_samples)
        sym_val_res = 0.
        avg_output = [0]*len(orginal_list)
        for i, data in enumerate(dataloaders[phase]):
            fname =  list(data[1])[0]
            batch_size =  len(data[0]['image'])
            extra_res =0.
            for j,orginal_img_path in enumerate(orginal_img_path_list):
                output = cal_sym_func(record_path,fname,orginal_img_path)
                extra_tmp = compute_sym_metric(orginal_list[j],output)
                extra_res += extra_tmp
                avg_output[j] += output
            sym_val_res += extra_res * batch_size
            records_sym_val_np[i] = extra_res
            print("id {} and current pair name is : {}".format(i,data[1]))
            print('the current sym val is {}'.format(extra_res))
            print('the current average sym val is {}'.format(sym_val_res/(i+1)/batch_size))
        avg_output = [res/len(dataloaders[phase].dataset) for res in avg_output]
        for i,res in enumerate(avg_output):
            sitk.WriteImage(sitk.GetImageFromArray(res),os.path.join(record_path,'avg_image_'+str(i)+'.nii.gz'))
        sym_val_res = sym_val_res/len(dataloaders[phase].dataset)
        print("the average {}_ sym val: {}  :".format(phase, sym_val_res))
        np.save(os.path.join(record_path,task_name+'records_sym'),records_sym_val_np)
