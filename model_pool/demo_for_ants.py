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






def read_nifty_reg_affine(affine_txt):
    res = np.loadtxt(affine_txt, delimiter=' ')
    matrix = res[:3,:3]
    matrix_cp = matrix.copy()
    matrix[0,2]=-matrix[0,2]
    matrix[1,2]=-matrix[1,2]
    matrix[2,0]=-matrix[2,0]
    matrix[2,1]=-matrix[2,1]
    matrix[0,0]= matrix_cp[0,0]
    matrix[1,1]= matrix_cp[1,1]
    trans = res[:3,3]
    trans_cp = trans.copy()
    trans[1] =-trans_cp[1]
    trans[0] =-trans_cp[0]
    return matrix, trans


moving_img_path = '/playpen/zyshen/debugs/demons/moving.nii.gz'
target_img_path = '/playpen/zyshen/debugs/demons/target.nii.gz'
ml_path = '/playpen/zyshen/debugs/demons/l_moving.nii.gz'
tl_path = '/playpen/zyshen/debugs/demons/l_target.nii.gz'

af_warped_path = '/playpen/zyshen/debugs/af_warped_img.nii.gz'

af_warped_lpath = '/playpen/zyshen/debugs/af_warped_label.nii.gz'
syn_warped_path = '/playpen/zyshen/debugs/syn_warped_img.nii.gz'
syn_warped_path2 = '/playpen/zyshen/debugs/syn_warped_img2.nii.gz'
syn_warped_lpath = '/playpen/zyshen/debugs/syn_warped_label.nii.gz'




from model_pool.nifty_reg_utils import nifty_reg_affine, nifty_reg_resample
import subprocess
# moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image.nii.gz'
# target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image.nii.gz'
af_pth = '/playpen/zyshen/debugs/nifty_affine/affine.txt'
af_nifty_warped_path =  '/playpen/zyshen/debugs/nifty_affine/af_nifty_warped_img.nii.gz'
af_nifty_warped2_path =  '/playpen/zyshen/debugs/nifty_affine/af_nifty_warped2_img.nii.gz'
af_sitk_warped_path = '/playpen/zyshen/debugs/nifty_affine/af_sitk_warped_img.nii.gz'
#
# cmd = nifty_reg_affine(ref=target_img_path, flo=moving_img_path, aff=af_pth, res=af_nifty_warped_path)
cmd = '\n' + nifty_reg_resample(ref=target_img_path,flo=moving_img_path,trans=af_pth, res=af_nifty_warped2_path)
process = subprocess.Popen(cmd, shell=True)
process.wait()


moving = sitk.ReadImage(moving_img_path)
target = sitk.ReadImage(target_img_path)

dimension=3
matrix, trans=read_nifty_reg_affine(af_pth)
affine = sitk.AffineTransform(dimension)
affine.SetMatrix(matrix.ravel())
affine.SetTranslation(trans)
#affine = affine.GetInverse()
initial_displacement_field = sitk.TransformToDisplacementField(affine,
                                                                sitk.sitkVectorFloat64,
                                                               target.GetSize(),
                                                               target.GetOrigin(),
                                                               target.GetSpacing(),
                                                               target.GetDirection())
tx = sitk.DisplacementFieldTransform(initial_displacement_field)
from model_pool.demons_utils_old import sitk_grid_sampling

warped_img = sitk_grid_sampling(target,moving, tx,
                                    is_label=False)
sitk.WriteImage(warped_img,af_sitk_warped_path)


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

# phi_ants = ants.create_warped_grid(moving, transform=syn_res['fwdtransforms'], grid_step=16, fixed_reference_image=target)
id_grid = ants.create_warped_grid(moving)
# phi_ants = ants.apply_transforms( fixed=id_grid, moving=id_grid, transformlist=syn_res['fwdtransforms'] )
#phi_ants = ants.image_read(syn_res['fwdtransforms'][1])
from model_pool.nifty_reg_utils import *
phi = ants.image_read(syn_res['fwdtransforms'][1])
id_grid =  phi
id_grid.dimension= 4
phi_ants = ants.apply_transforms( fixed=target, moving=moving, transformlist=syn_res['fwdtransforms'])
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
