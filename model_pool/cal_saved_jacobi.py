from time import time

from model_pool.nifty_reg_utils import nifty_reg_resample
from pipLine.utils import *
import os
import SimpleITK as sitk
import ants
import subprocess


def compute_jacobi_map(jacobian):
    jacobi_abs = - np.sum(jacobian[jacobian < 0.])  #
    jacobi_num = np.sum(jacobian < 0.)
    print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
    print("the number of fold points for current batch is {}".format(jacobi_num))
    # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
    jacobi_abs_mean = jacobi_abs  # / np.prod(map.shape)
    return jacobi_abs_mean,jacobi_num



def cal_ants_jacobi(shared_reference_path,record_path,fname):
    disp_pth =os.path.join(record_path,fname+'_disp.nii.gz')
    mat_pth = os.path.join(record_path,fname+"_affine.mat")
    reference = ants.image_read(shared_reference_path)
    warp_tmp = ants.apply_transforms(fixed=reference, moving=reference,
                                         transformlist=[disp_pth,mat_pth], compose=record_path)
    jacobian = ants.create_jacobian_determinant_image(reference, warp_tmp, False)
    jacobian_np = jacobian.numpy()
    jacobian_np[jacobian_np==0]=-1
    return jacobian_np

def cal_demons_jacobi(shared_reference_path,record_path,fname):
    # af_txt = os.path.join(record_path,fname+'_af.txt')
    # output_pth = os.path.join(record_path,'tmp_output.nii.gz')
    # cmd = nifty_reg_resample(ref=shared_reference_path, flo=shared_reference_path, trans=af_txt, res=output_pth, inter=0)
    # process = subprocess.Popen(cmd, shell=True)
    # process.wait()
    disp_pth =os.path.join(record_path,fname+'_disp.nii.gz')
    disp = sitk.ReadImage(disp_pth)
    jacobi_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobi_image = jacobi_filter.Execute(disp)
    jacobian_np = sitk.GetArrayFromImage(jacobi_image)
    return jacobian_np



def cal_jacobi(opt,dataloaders,task_name=''):
    record_path = opt['tsk_set']['path']['record_path']
    model_name = opt['tsk_set']['model']
    phases=['test']
    reference_image_path = '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_affine_jacobi/records/target.nii.gz'
    cal_jacobi_func = None
    if model_name == 'ants':
        cal_jacobi_func = cal_ants_jacobi
    if model_name == 'demons':
        cal_jacobi_func = cal_demons_jacobi



    for phase in phases:
        num_samples = len(dataloaders[phase])
        records_jacobi_val_np = np.zeros(num_samples)
        records_jacobi_num_np = np.zeros(num_samples)
        jacobi_val_res = 0.
        jacobi_num_res = 0.
        for i, data in enumerate(dataloaders[phase]):
            fname =  list(data[1])[0]
            batch_size =  len(data[0]['image'])
            jacobian_np = cal_jacobi_func(reference_image_path, record_path,fname)
            extra_res = compute_jacobi_map(jacobian_np)
            jacobi_val_res += extra_res[0] * batch_size
            jacobi_num_res += extra_res[1] * batch_size
            records_jacobi_val_np[i] = extra_res[0]
            records_jacobi_num_np[i] = extra_res[1]
            print("id {} and current pair name is : {}".format(i,data[1]))
            print('the current averge jocobi val is {}'.format(jacobi_val_res/(i+1)/batch_size))
            print('the current averge jocobi num is {}'.format(jacobi_num_res/(i+1)/batch_size))
        jacobi_val_res = jacobi_val_res/len(dataloaders[phase].dataset)
        jacobi_num_res = jacobi_num_res/len(dataloaders[phase].dataset)
        print("the average {}_ jacobi val: {}  :".format(phase, jacobi_val_res))
        print("the average {}_ jacobi num: {}  :".format(phase, jacobi_num_res))
        np.save(os.path.join(record_path,task_name+'records_jacobi'),records_jacobi_val_np)
        np.save(os.path.join(record_path,task_name+'records_jacobi_num'),records_jacobi_num_np)


