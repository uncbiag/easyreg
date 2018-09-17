import os
import numpy as np
import ants
import time
import SimpleITK as sitk
transform_types = {'SynBold',
                                'SynBoldAff',
                                'ElasticSyn',
                                'SyN',
                                'SyNRA',
                                'SyNOnly',
                                'SyNAggro',
                                'SyNCC',
                                'TRSAA',
                                'SyNabp',
                                'SyNLessAggro',
                                'TVMSQ',
                                'TVMSQC',
                                'Rigid',
                                'Similarity',
                                'Translation',
                                'Affine',
                                'AffineFast',
                                'BOLDAffine',
                                'QuickRigid',
                                'DenseRigid',
'BOLDRigid'}


moving_img_path = '/playpen/zyshen/debugs/9002116_20050715_SAG_3D_DESS_RIGHT_10423916_image.nii.gz'
target_img_path = '/playpen/zyshen/debugs/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_image.nii.gz'
ml_path = '/playpen/zyshen/debugs/9002116_20050715_SAG_3D_DESS_RIGHT_10423916_prediction_step1_batch6_16_reflect.nii.gz'
tl_path = '/playpen/zyshen/debugs/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_prediction_step1_batch6_16_reflect.nii.gz'

af_warped_path = '/playpen/zyshen/debugs/af_warped_img.nii.gz'
af_warped_lpath = '/playpen/zyshen/debugs/af_warped_label.nii.gz'
syn_warped_path = '/playpen/zyshen/debugs/syn_warped_img.nii.gz'
syn_warped_lpath = '/playpen/zyshen/debugs/syn_warped_label.nii.gz'

moving = ants.image_read(moving_img_path)
target = ants.image_read(target_img_path)
ml_sitk = sitk.ReadImage(ml_path)
tl_sitk = sitk.ReadImage(tl_path)
ml_np = sitk.GetArrayFromImage(ml_sitk)
tl_np = sitk.GetArrayFromImage(tl_sitk)
l_moving =ants.from_numpy(np.transpose(ml_np),spacing=moving.spacing,direction=moving.direction,origin=moving.origin)
l_target =ants.from_numpy(np.transpose(tl_np),spacing=target.spacing,direction=target.direction,origin=target.origin)


start = time.time()
#
# affine_file = ants.affine_initializer( target, moving )
# af_img = ants.apply_transforms(fixed=target, moving=moving,transformlist=affine_file)
#
# af_label = ants.apply_transforms(fixed=l_target, moving=l_moving,transformlist=affine_file,interpolator='nearestNeighbor')
# ants.image_write(af_img,af_warped_path)
# ants.image_write(af_label,af_warped_lpath)


syn_res = ants.registration(fixed=target, moving=moving, type_of_transform ='SyN')
syn_label = ants.apply_transforms(fixed=l_target, moving=l_moving,
                                      transformlist=syn_res['fwdtransforms'],interpolator='nearestNeighbor') #interpolator

ants.image_write(syn_res['warpedmovout'],syn_warped_path)
ants.image_write(syn_label,syn_warped_lpath)

phi_ants = ants.create_warped_grid(moving, transform=syn_res['fwdtransforms'], grid_step=16, fixed_reference_image=target)
id_grid = ants.create_warped_grid(moving)
phi_ants = ants.apply_transforms( fixed=target, moving=id_grid, transformlist=syn_res['fwdtransforms'] )
phi_ants = ants.image_read(syn_res['fwdtransforms'][1])
phi_np = np.transpose(phi_ants.numpy(),(3,0,1,2))
print('registration finished and takes: :', time.time() - start)
