import os
import numpy as np
import ants
import time
import SimpleITK as sitk
from .nifty_reg_utils import expand_batch_ch_dim
import subprocess



def performAntsRegistration(param, mv_path, target_path, registration_type='syn', record_path = None, ml_path=None,tl_path= None, fname = None):
    """
    call [AntsPy](https://github.com/ANTsX/ANTsPy),

    :param param: ParameterDict, affine related params
    :param mv_path: path of moving image
    :param target_path: path of target image
    :param registration_type: type of registration, support 'affine' and 'syn'(include affine)
    :param record_path: path of saving results
    :param ml_path: path of label of moving image
    :param tl_path: path of label fo target image
    :param fname: pair name or saving name of the image pair
    :return: warped image, warped label, transformation map (None), jacobian map
    """
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
    # #disp = nifty_read_phi(syn_res['fwdtransforms'][0])
    # #disp = np.transpose(disp, (0,1,4, 3, 2))
    # composed_transform = ants.apply_transforms(fixed=target, moving=moving,
    #                                       transformlist=syn_res['fwdtransforms'],compose= record_path)
    # cmd = 'mv ' + composed_transform + ' ' + os.path.join(record_path,fname+'_disp.nii.gz')
    # composed_inv_transform = ants.apply_transforms(fixed=target, moving=moving,
    #                                       transformlist=syn_res['invtransforms'],compose= record_path)
    # cmd = 'mv ' + composed_inv_transform + ' ' + os.path.join(record_path,fname+'_invdisp.nii.gz')
    cmd = 'mv ' + syn_res['fwdtransforms'][0] + ' ' + os.path.join(record_path,fname+'_disp.nii.gz')
    cmd += '\n mv ' + syn_res['fwdtransforms'][1] + ' ' + os.path.join(record_path,fname+'_affine.mat')
    cmd += '\n mv ' + syn_res['invtransforms'][0] + ' ' + os.path.join(record_path,fname+'_invdisp.nii.gz')
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    jacobian_np = None
    if registration_type =='syn':
        jacobian = ants.create_jacobian_determinant_image(target, os.path.join(record_path,fname+'_disp.nii.gz'), False)
        jacobian_np = jacobian.numpy()

    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), phi, jacobian_np
