import os
import SimpleITK as sitk
from mermaid.utils import compute_warped_image_multiNC
from easyreg.reg_data_utils import get_file_name
from tools.image_rescale import save_image_with_given_reference
from glob import glob
import numpy as np
import torch
from functools import reduce


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3),labels.size(4)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = target

    return target

def compute_atlas_label(lsource_folder_path, to_atlas_folder_pth,atlas_type, atlas_to_l_switcher,output_folder):
    to_atlas_pth_list = glob(os.path.join(to_atlas_folder_pth,"*"+atlas_type))[:100]
    to_atlas_name_list = [get_file_name(to_atlas_pth) for to_atlas_pth in to_atlas_pth_list]
    l_pth_list = [os.path.join(lsource_folder_path,name.replace(*atlas_to_l_switcher)+'.nii.gz') for name in to_atlas_name_list]
    fr_sitk = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    l_list = [fr_sitk(pth)[None] for pth in l_pth_list]
    to_atlas_list = [np.transpose(fr_sitk(pth)) for pth in to_atlas_pth_list]
    l = np.stack(l_list).astype(np.int64)
    num_c = len(np.unique(l))
    to_atlas = np.stack(to_atlas_list)
    l= torch.LongTensor(l)
    to_atlas = torch.Tensor(to_atlas)
    l_onehot = make_one_hot(l,C=num_c)
    spacing = 1./(np.array(l.shape[2:])-1)
    l_onehot = l_onehot.to(torch.float32)
    warped_one_hot = compute_warped_image_multiNC(l_onehot,to_atlas,spacing = spacing, spline_order=1,zero_boundary=True)
    sum_one_hot = torch.sum(warped_one_hot,0,keepdim=True)
    voting = torch.max(torch.Tensor(sum_one_hot), 1)[1][None].to(torch.float32)
    save_image_with_given_reference(voting,[l_pth_list[0]],output_folder,["atlas_label"])
atlas_type = "_atlas_image_phi.nii.gz"
atlas_to_l_switcher = ("image_atlas_image_phi","masks")
lsource_folder_path ="/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
to_atlas_folder_pth ="/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/test_sm/reg/res/records"
output_folder = "/playpen-raid/zyshen/data/oai_seg/atlas"
compute_atlas_label(lsource_folder_path, to_atlas_folder_pth,atlas_type, atlas_to_l_switcher,output_folder)
