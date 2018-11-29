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


moving_img_path = '/playpen/zyshen/debugs/demons/moving.nii.gz'
target_img_path = '/playpen/zyshen/debugs/demons/target.nii.gz'
ml_path = '/playpen/zyshen/debugs/demons/l_moving.nii.gz'
tl_path = '/playpen/zyshen/debugs/demons/l_target.nii.gz'

af_warped_path = '/playpen/zyshen/debugs/af_warped_img.nii.gz'
af_warped_lpath = '/playpen/zyshen/debugs/af_warped_label.nii.gz'
syn_warped_path = '/playpen/zyshen/debugs/syn_warped_img.nii.gz'
syn_warped_path2 = '/playpen/zyshen/debugs/syn_warped_img2.nii.gz'
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

syn_res = ants.registration(fixed=target, moving=moving, type_of_transform ='SyN')
if 'GenericAffine.mat' in syn_res['fwdtransforms'][0]:
    tmp1 = syn_res['fwdtransforms'][0]
    tmp2 = syn_res['fwdtransforms'][1]
    syn_res['fwdtransforms'][0] = tmp2
    syn_res['fwdtransforms'][1] = tmp1
syn_warp_tmp = ants.apply_transforms(fixed=target, moving=moving,
                                      transformlist=syn_res['fwdtransforms'],compose= '/playpen/zyshen/debugs/')
syn_warp_tmp2 = ants.apply_transforms(fixed=target, moving=moving,
                                      transformlist=syn_warp_tmp)
syn_label = ants.apply_transforms(fixed=l_target, moving=l_moving,
                                      transformlist=syn_res['fwdtransforms'],interpolator='nearestNeighbor') #interpolator
jacobian = ants.create_jacobian_determinant_image(target,syn_res['fwdtransforms'][0],False)

ants.image_write(syn_res['warpedmovout'],syn_warped_path)
ants.image_write(syn_warp_tmp,syn_warped_path2)

ants.image_write(syn_label,syn_warped_lpath)
print('registration finished and takes: :', time.time() - start)