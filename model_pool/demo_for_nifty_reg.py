import sys
import subprocess
import os
import numpy as np
from model_pool.nifty_reg_utils import *
#from mermaid.pyreg.utils import identity_map_multiN
#
mv_path = ''
target_path = ''
registration_type = 'bspline'
moving_img_path = '/playpen/zyshen/debugs/demons/moving.nii.gz'
target_img_path = '/playpen/zyshen/debugs/demons/target.nii.gz'
moving_label_path ='/playpen/zyshen/debugs/demons/l_moving.nii.gz'
# moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_image.nii.gz'
# target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_image.nii.gz'
# moving_label_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_label_all.nii.gz'

_,_, phi=performRegistration(moving_img_path,target_img_path,registration_type,record_path=None,ml_path=moving_label_path)



#
phi = nifty_read_phi('./deformation.nii.gz')
disp = nifty_read_phi('./displacement.nii')
spacing = 1. / (np.array(phi.shape[2:]) - 1)
sz = phi.shape[2:]
identity_map = np.zeros([1, 3, sz[0],sz[1],sz[2]],dtype=np.float32)
identity_map[0] = np.mgrid[0:sz[0],0:sz[1],0:sz[2]]
idd = phi - disp

print("done")
import mermaid.pyreg.finite_differences as  fdt
fd = fdt.FD_np(np.array([1.,1.,1.]))
dfx= fd.dXf(phi[:, 0, ...])
dfy= fd.dYf(phi[:, 1, ...])
dfz= fd.dZf(phi[:, 2, ...])
#jacobi_abs = np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
jacobi_abs = np.sum(dfx<0.) + np.sum(dfy<0.) + np.sum(dfz<0.)
print(jacobi_abs)


print("done")
import mermaid.pyreg.finite_differences as  fdt
fd = fdt.FD_np(np.array([1.,1.,1.]))
dfx= fd.dXf(disp[:, 0, ...])
dfy= fd.dYf(disp[:, 1, ...])
dfz= fd.dZf(disp[:, 2, ...])
#jacobi_abs = np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
jacobi_abs = np.sum(-dfx+1<0) + np.sum(-dfy+1<0) + np.sum(dfz+1<0)
print(jacobi_abs)