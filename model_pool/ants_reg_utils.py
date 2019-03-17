import os
import numpy as np
import ants
import time
import SimpleITK as sitk
from model_pool.global_variable import param_in_ants
from model_pool.nifty_reg_utils import expand_batch_ch_dim
from mermaid.pyreg.utils import identity_map_multiN
import subprocess
import nibabel as nib



def nifty_read_phi(path):
    phi_nib = nib.load(path)
    phi = phi_nib.get_fdata()
    phi_tmp = np.zeros([1,3]+list(phi.shape[:3]))
    phi_tmp[0,0] = -phi[...,0,0]
    phi_tmp[0,1] = -phi[...,0,2]
    phi_tmp[0,2] =  phi[...,0,1]
    return phi_tmp


def __init_identity_map(moving,spacing):
    spacing = 1. / (np.array(moving.shape) - 1)
    identity_map = identity_map_multiN(moving.shape, spacing)
    phi_tmp = np.zeros( list(identity_map.shape[2:])+[3])
    phi_tmp[...,0] = identity_map[ 0, 0,...]
    phi_tmp[...,1] = identity_map[0, 1,...]
    phi_tmp[..., 2] = identity_map[0, 2,...]
    return phi_tmp




def performAntsRegistration(mv_path, target_path, registration_type='syn', record_path = None, ml_path=None,tl_path= None, fname = None, return_syn=False):
    loutput =None
    phi = None
    moving = ants.image_read(mv_path)
    target = ants.image_read(target_path)
    if ml_path is not None:
        ml_sitk = sitk.ReadImage(ml_path)
        tl_sitk = sitk.ReadImage(tl_path)
        ml_np = sitk.GetArrayFromImage(ml_sitk)
        tl_np = sitk.GetArrayFromImage(tl_sitk)
        l_moving = ants.from_numpy(np.transpose(ml_np), spacing=moving.spacing, direction=moving.direction,
                                   origin=moving.origin)
        l_target = ants.from_numpy(np.transpose(tl_np), spacing=target.spacing, direction=target.direction,
                                   origin=target.origin)


    start = time.time()
    if registration_type =='affine':
        affine_file = ants.affine_initializer(target, moving)
        af_img = ants.apply_transforms(fixed=target, moving=moving, transformlist=affine_file)
        if ml_path is not None:
            loutput = ants.apply_transforms(fixed=l_target, moving=l_moving, transformlist=affine_file,
                                         interpolator='nearestNeighbor')
            loutput = loutput.numpy()
        output = af_img.numpy()
        print('affine registration finished and takes: :', time.time() - start)
    #print("param_in_ants:{}".format(param_in_ants))
    if registration_type =='syn':
        syn_res = ants.registration(fixed=target, moving=moving, type_of_transform='SyNCC', grad_step=0.2,
                                    flow_sigma=3,  # intra 3
                                    total_sigma=0.1,
                                    aff_metric='mattes',
                                    aff_sampling=8,
                                    syn_metric='mattes',
                                    syn_sampling=32,
                                    reg_iterations=(80, 50, 20))

        print(syn_res['fwdtransforms'])
        if 'GenericAffine.mat' in syn_res['fwdtransforms'][0]:
            tmp1 = syn_res['fwdtransforms'][0]
            tmp2 = syn_res['fwdtransforms'][1]
            syn_res['fwdtransforms'][0] = tmp2
            syn_res['fwdtransforms'][1] = tmp1
        if ml_path is not None:
            time.sleep(1)

            loutput = ants.apply_transforms(fixed=l_target, moving=l_moving,
                                          transformlist=syn_res['fwdtransforms'],
                                          interpolator='nearestNeighbor')
            loutput = loutput.numpy()
        output = syn_res['warpedmovout'].numpy()
        print('syn registration finished and takes: :', time.time() - start)


    output = np.transpose(output,(2,1,0))
    loutput = np.transpose(loutput,(2,1,0)) if loutput is not None else None
    disp = nifty_read_phi(syn_res['fwdtransforms'][0])
    disp = np.transpose(disp, (0,1,4, 3, 2))
    cmd = 'mv ' + syn_res['fwdtransforms'][0] + ' ' + os.path.join(record_path,fname+'_disp.nii.gz')
    cmd += '\n mv ' + syn_res['fwdtransforms'][1] + ' ' + os.path.join(record_path,fname+'_affine.mat')
    cmd += '\n mv ' + syn_res['invtransforms'][0] + ' ' + os.path.join(record_path,fname+'_invdisp.nii.gz')
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    jacobian_np = None
    if registration_type =='syn':
        jacobian = ants.create_jacobian_determinant_image(target, os.path.join(record_path,fname+'_disp.nii.gz'), False)
        jacobian_np = jacobian.numpy()

    if not return_syn:
        return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), disp,jacobian_np
    else:
        return syn_res