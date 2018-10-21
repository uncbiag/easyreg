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
#
# affine_file = ants.affine_initializer( target, moving )
# af_img = ants.apply_transforms(fixed=target, moving=moving,transformlist=affine_file)
#
# af_label = ants.apply_transforms(fixed=l_target, moving=l_moving,transformlist=affine_file,interpolator='nearestNeighbor')
# ants.image_write(af_img,af_warped_path)
# ants.image_write(af_label,af_warped_lpath)


syn_res = ants.registration(fixed=target, moving=moving, type_of_transform ='SyN')
syn_warp_tmp = ants.apply_transforms(fixed=target, moving=moving,
                                      transformlist=syn_res['fwdtransforms'])
syn_label = ants.apply_transforms(fixed=l_target, moving=l_moving,
                                      transformlist=syn_res['fwdtransforms'],interpolator='nearestNeighbor') #interpolator

ants.image_write(syn_res['warpedmovout'],syn_warped_path)
ants.image_write(syn_warp_tmp,syn_warped_path2)

ants.image_write(syn_label,syn_warped_lpath)
print('registration finished and takes: :', time.time() - start)

# phi_ants = ants.create_warped_grid(moving, transform=syn_res['fwdtransforms'], grid_step=16, fixed_reference_image=target)
id_grid = ants.create_warped_grid(moving)
# phi_ants = ants.apply_transforms( fixed=id_grid, moving=id_grid, transformlist=syn_res['fwdtransforms'] )
#phi_ants = ants.image_read(syn_res['fwdtransforms'][1])
from model_pool.nifty_reg_utils import *
phi = ants.image_read(syn_res['fwdtransforms'][1])
id_grid =  phi
id_grid.dimension= 4
phi_ants = ants.apply_transforms( fixed=target, moving=id_grid, transformlist=syn_res['fwdtransforms'])
ants.image_write(phi_ants,'/playpen/zyshen/debugs/phi_ants.nii.gz')
phi_warped = ants.image_read('/playpen/zyshen/debugs/phi_ants.nii.gz')
print(phi_warped.numpy().shape)

print("done")
import mermaid.pyreg.finite_differences as  fdt
fd = fdt.FD_np(np.array([1.,1.,1.]))
dfx= fd.dXf(phi[:, 0, ...])
dfy= fd.dYf(phi[:, 1, ...])
dfz= fd.dZf(phi[:, 2, ...])
#jacobi_abs = np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
jacobi_abs = np.sum(dfx+1<0.) + np.sum(dfy+1<0.) + np.sum(dfz+1<0.)
print(jacobi_abs)
