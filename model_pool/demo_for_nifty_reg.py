import sys
import subprocess
import os
import numpy as np
from model_pool.nifty_reg_utils import *
#from mermaid.pyreg.utils import identity_map_multiN
import ants

def __read_and_clean_itk_info(path):
    return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))

def resize_input_img_and_save_it_as_tmp(img_pth, is_label=False,fname=None,debug_path=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized  # (80,192,192)
        :return:
        """
        img_org = sitk.ReadImage(img_pth)
        img = __read_and_clean_itk_info(img_pth)
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud([0.8,0.9,0.8])
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        fpth = os.path.join(debug_path,fname)
        # img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        # img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        # img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth



mv_path = ''
target_path = ''
registration_type = 'bspline'
record_path = '/playpen/zyshen/debugs/nifty_reg'
if not os.path.exists(record_path):
    os.mkdir(record_path)
moving_img_path = '/playpen/zyshen/debugs/zhengyang/moving_0.nii.gz'
target_img_path = '/playpen/zyshen/debugs/zhengyang/target_0.nii.gz'
moving_label_path = '/playpen/zyshen/debugs/zhengyang/l_moving0.nii.gz'
target_label_path = '/playpen/zyshen/debugs/zhengyang/l_target0.nii.gz'


# moving_img_path = resize_input_img_and_save_it_as_tmp(moving_img_path,is_label=False,fname='moving.nii.gz',debug_path=record_path)
# target_img_path = resize_input_img_and_save_it_as_tmp(target_img_path,is_label=False,fname='target.nii.gz',debug_path=record_path)
# moving_label_path = resize_input_img_and_save_it_as_tmp(moving_label_path,is_label=True,fname='l_moving.nii.gz',debug_path=debug_path)
# target_label_path = resize_input_img_and_save_it_as_tmp(target_label_path,is_label=True,fname='l_target.nii.gz',debug_path=debug_path)














# moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_image.nii.gz'
# target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_image.nii.gz'
# moving_label_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_label_all.nii.gz'
_,_, phi,jacobi=performRegistration(moving_img_path,target_img_path,registration_type,record_path=record_path,ml_path=moving_label_path)

#
phi = nifty_read_phi(os.path.join(record_path,'deformation.nii.gz'))
disp = nifty_read_phi(os.path.join(record_path,'displacement.nii.gz'))
spacing = 1. / (np.array(phi.shape[2:]) - 1)
sz = phi.shape[2:]
identity_map = np.zeros([1, 3, sz[0],sz[1],sz[2]],dtype=np.float32)
identity_map[0] = np.mgrid[0:sz[0],0:sz[1],0:sz[2]]
idd = phi - disp

print("done")
import mermaid.pyreg.finite_differences as  fdt
fd = fdt.FD_np(np.array([1.,1.,1.]))
dfx= fd.dXb(phi[:, 0, ...])
dfy= fd.dYb(phi[:, 1, ...])
dfz= fd.dZb(phi[:, 2, ...])
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