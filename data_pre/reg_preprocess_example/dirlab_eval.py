import os
import numpy as np
import torch
import mermaid.utils as py_utils
from data_pre.reg_preprocess_example.vtk_utils import save_vtk
from tools.visual_tools import save_3D_img_from_numpy

COPD_ID=[
   "copd_000001",
   "copd_000002",
   "copd_000003",
   "copd_000004",
   "copd_000005",
   "copd_000006",
   "copd_000007",
   "copd_000008",
   "copd_000009",
   "copd_000010"
]

dirlab_folder_path = "/playpen-raid1/zyshen/data/lung_resample_350/landmarks"
org_image_sz = np.array([350,350,350])                 #physical spacing equals to 1
def read_index(index_path):
    index = np.load(index_path)
    return flip_coord(index)

def flip_coord(coord):
    coord_flip = coord.copy()
    coord_flip[:,0] = coord[:,2]
    coord_flip[:,2] = coord[:,0]
    return coord_flip

def get_landmark(pair_name, resampled_image_sz):
    case_id = pair_name.split("_")
    case_id =case_id[0]+"_"+case_id[1]
    slandmark_index_path = os.path.join(dirlab_folder_path,case_id+"_EXP_index.npy")
    tlandmark_index_path = os.path.join(dirlab_folder_path,case_id+"_INSP_index.npy")
    slandmark_index = read_index(slandmark_index_path)
    tlandmark_index = read_index(tlandmark_index_path)
    slandmark_index = slandmark_index*(resampled_image_sz-1)/(org_image_sz-1)
    tlandmark_index = tlandmark_index*(resampled_image_sz-1)/(org_image_sz-1)
    physical_spacing = (org_image_sz-1)/(resampled_image_sz-1)
    return slandmark_index, tlandmark_index, physical_spacing


def eval_on_dirlab_per_case(forward_map,inv_map, pair_name,moving, target,record_path):
    transform_shape = np.array(forward_map.shape[2:])
    slandmark_index,tlandmark_index, physical_spacing = get_landmark(pair_name,transform_shape)
    spacing = 1./(transform_shape-1)
    slandmark_img_coord = spacing*slandmark_index
    tlandmark_img_coord = spacing*tlandmark_index
    # target = target.squeeze().clone()
    # for coord in tlandmark_index:
    #     coord_int  = [int(c) for c in coord]
    #     target[coord_int[0],coord_int[1],coord_int[2]] = 10.
    # save_3D_img_from_numpy(target.detach().cpu().numpy().squeeze(),"/playpen-raid2/zyshen/debug/{}_debug.nii.gz".format(pair_name))


    tlandmark_img_coord_reshape = torch.Tensor(tlandmark_img_coord.transpose(1,0)).view([1,3,-1,1,1])
    tlandmark_img_coord_reshape = tlandmark_img_coord_reshape.to(forward_map.device)
    ts_landmark_img_coord = py_utils.compute_warped_image_multiNC(forward_map, tlandmark_img_coord_reshape*2-1, spacing, 1, zero_boundary=False,use_01_input=False)
    ts_landmark_img_coord = ts_landmark_img_coord.squeeze().transpose(1,0).detach().cpu().numpy()
    diff_ts = (slandmark_img_coord - ts_landmark_img_coord)/spacing*physical_spacing

    # target = target.squeeze().clone()
    # for coord in ts_landmark_img_coord:
    #     coord_int  = [int(c) for c in coord/spacing]
    #     target[coord_int[0],coord_int[1],coord_int[2]] = 10.
    # save_3D_img_from_numpy(target.detach().cpu().numpy().squeeze(),"/playpen-raid2/zyshen/debug/{}_debug_warped.nii.gz".format(pair_name))

    slandmark_img_coord_reshape = torch.Tensor(slandmark_img_coord.transpose(1, 0)).view([1, 3, -1, 1, 1])
    slandmark_img_coord_reshape = slandmark_img_coord_reshape.to(inv_map.device)
    st_landmark_img_coord = py_utils.compute_warped_image_multiNC(inv_map, slandmark_img_coord_reshape * 2 - 1,
                                                                  spacing, 1, zero_boundary=False, use_01_input=False)
    st_landmark_img_coord = st_landmark_img_coord.squeeze().transpose(1, 0).detach().cpu().numpy()
    landmark_saving_folder = os.path.join(record_path,"landmarks")
    os.makedirs(landmark_saving_folder, exist_ok=True)
    save_vtk(os.path.join(landmark_saving_folder,"{}_source.vtk".format(pair_name)),{"points":slandmark_img_coord})
    save_vtk(os.path.join(landmark_saving_folder,"{}_target.vtk".format(pair_name)),{"points":tlandmark_img_coord})
    save_vtk(os.path.join(landmark_saving_folder,"{}_target_warp_to_source.vtk".format(pair_name)),{"points":ts_landmark_img_coord})
    save_vtk(os.path.join(landmark_saving_folder,"{}_source_warp_to_target.vtk".format(pair_name)),{"points":st_landmark_img_coord})
    diff_st = (tlandmark_img_coord - st_landmark_img_coord) / spacing * physical_spacing

    return np.linalg.norm(diff_ts,ord=2,axis=1).mean(), np.linalg.norm(diff_st,ord=2,axis=1).mean()

def eval_on_dirlab(forward_map,inverse_map, pair_name_list, moving, target, record_path=None):
    diff_ts, diff_st = [], []
    for _forward_map, _inv_map, pair_name, _moving, _target in zip(forward_map,inverse_map,pair_name_list, moving, target):
        if pair_name in COPD_ID:
            _diff_ts, _diff_st = eval_on_dirlab_per_case(_forward_map[None],_inv_map[None], pair_name,_moving,_target, record_path)
            diff_ts.append(_diff_ts), diff_st.append(_diff_st)
    print("dirlab landmark source to target error:{}:{}".format(pair_name_list,diff_st))
    print("dirlab landmark target_to source error:{}:{}".format(pair_name_list,diff_ts))
