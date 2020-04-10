import nibabel as nib
import numpy as np

def transform_disp_into_torch_form(inv_transform_file):
    inv_map = nib.load(inv_transform_file)
    inv_map = inv_map.get_fdata()
    assert inv_map.shape[0]==3
    inv_map = np.transpose(inv_map,[3,2,1,0])
    return inv_map
