import numpy as np
import sys,os
import SimpleITK as sitk
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
from mermaid.pyreg.utils import identity_map_multiN
img_size = [80,192,192]
spacing = 1. / (np.array(img_size) - 1)
identity_map = identity_map_multiN([1,1]+img_size, spacing)
print(identity_map.shape)
id_path  =  './identity.nii.gz'

sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,0]),id_path.replace('identity','identity_x'))
sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,1]),id_path.replace('identity','identity_y'))
sitk.WriteImage(sitk.GetImageFromArray(identity_map[0,2]),id_path.replace('identity','identity_z'))
x = sitk.ReadImage('./identity_x.nii.gz')
y = sitk.ReadImage('./identity_y.nii.gz')
z = sitk.ReadImage('./identity_z.nii.gz')
print(sitk.GetArrayFromImage(x).shape,sitk.GetArrayFromImage(y).shape,sitk.GetArrayFromImage(z).shape)
