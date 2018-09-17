import os
import numpy as np
import ants
import time
import SimpleITK as sitk

from model_pool.nifty_reg_utils import expand_batch_ch_dim


def performAntsRegistration(mv_path, target_path, registration_type='syn', record_path = None, ml_path=None,tl_path= None):
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

    if registration_type =='syn':
        syn_res = ants.registration(fixed=target, moving=moving, type_of_transform='SyN')
        if ml_path is not None:
            loutput = ants.apply_transforms(fixed=l_target, moving=l_moving,
                                          transformlist=syn_res['fwdtransforms'],
                                          interpolator='nearestNeighbor')
            loutput = loutput.numpy()
        output = syn_res['warpedmovout'].numpy()
        #phi_ants = ants.create_warped_grid(moving, transform=affine_file, fixed_reference_image=target)
        # phi = np.transpose(phi_ants.numpy(), (3, 2, 1, 0))
        print('syn registration finished and takes: :', time.time() - start)


    output = np.transpose(output,(2,1,0))
    loutput = np.transpose(loutput,(2,1,0))



    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput),None# np.expand_dims(phi,0)