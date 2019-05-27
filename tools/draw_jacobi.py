import numpy as np
from mermaid.pyreg.viewers import *
import SimpleITK as sitk
sz  = np.array([80,192,192])
disp_pth = '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/visualize_affine/records/3D/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image_9357383_20040927_SAG_3D_DESS_LEFT_016610250606_imagemap.nii.gz'
img_pth = '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/visualize_affine/records/3D/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image_9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image_reproduce.nii.gz'
disp = sitk.ReadImage(disp_pth)
disp = sitk.GetArrayFromImage(disp)
img  = sitk.GetArrayFromImage(sitk.ReadImage(img_pth))
#disp = np.transpose(disp,(3,2,1,0))


spacing = 1. / (sz - 1)
identity_map = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
grid = identity_map+ disp
grid[0] = grid[0]*spacing[0]
grid[1] = grid[1]*spacing[1]
grid[2] = grid[2]*spacing[2]
grid = grid*2-1
print(np.max(grid), np.min(grid))


fig,ax = plt.subplots(1,3,figsize=(50, 30))
img = np.zeros_like(img)
img[1,:,1]=1
plt.setp(plt.gcf(), 'facecolor', 'white')
plt.style.use('grayscale')

ivx = ImageViewer3D_Sliced_Contour( ax[0], img,grid, 0, '',showColorbar=True)
ivy = ImageViewer3D_Sliced_Contour( ax[1], img,grid, 1, '',showColorbar=True)
ivz = ImageViewer3D_Sliced_Contour( ax[2], img,grid, 2, '',showColorbar=True)

plt.clim(vmin=-1., vmax=1.)
plt.show()
