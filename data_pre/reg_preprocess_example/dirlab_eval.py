import os
import json
import numpy as np
import torch
import SimpleITK as sitk
import pyvista as pv
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


COPD_info = {"copd_000001": {"insp":{'size': [512, 512, 482],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -310.625]},
                        "exp":{'size': [512, 512, 473],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -305.0]}},
              "copd_000002":  {"insp":{'size': [512, 512, 406],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-176.9, -165.0, -254.625]},
                        "exp":{'size': [512, 512, 378],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-177.0, -165.0, -237.125]}},
              "copd_000003":  {"insp":{'size': [512, 512, 502],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -343.125]},
                        "exp":{'size': [512, 512, 464],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -319.375]}},
              "copd_000004":  {"insp":{'size': [512, 512, 501],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -308.25]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -283.25]}},
              "copd_000005":  {"insp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]},
                        "exp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]}},
              "copd_000006":  {"insp":{'size': [512, 512, 474],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -299.625]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -291.5]}},
              "copd_000007":  {"insp":{'size': [512, 512, 446],'spacing': [0.625, 0.625, 0.625], 'origin': [-150.7, -160.0, -301.375]},
                        "exp":{'size': [512, 512, 407],'spacing': [0.625, 0.625, 0.625], 'origin': [-151.0, -160.0, -284.25]}},
              "copd_000008":  {"insp":{'size': [512, 512, 458],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -313.625]},
                        "exp":{'size': [512, 512, 426],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -294.625]}},
              "copd_000009":  {"insp":{'size': [512, 512, 461],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -310.25]},
                        "exp":{'size': [512, 512, 380],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -259.625]}},
              "copd_000010": {"insp":{'size': [512, 512, 535],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -355.0]},
                        "exp":{'size': [512, 512, 539],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -346.25]}}
              }


# resampled_image_json_info = "/playpen-raid1/zyshen/data/lung_resample_350/dirlab_350_sampled.json"
# dirlab_folder_path = "/playpen-raid1/zyshen/data/lung_resample_350/landmarks"
# org_image_sz = np.array([350,350,350])                 #physical spacing equals to 1
# with open(resampled_image_json_info) as f:
#     img_info = json.load(f)
#
#
# def read_index(index_path):
#     index = np.load(index_path)
#     return flip_coord(index)
#
# def get_spacing_and_origin(case_id):
#     return np.flip(img_info[case_id]["spacing"]), np.flip(img_info[case_id]["origin"])
#
# def flip_coord(coord):
#     coord_flip = coord.copy()
#     coord_flip[:,0] = coord[:,2]
#     coord_flip[:,2] = coord[:,0]
#     return coord_flip
#
# def transfer_into_itk_coord(points):
#     return flip_coord(points)
#
# def get_landmark(pair_name, resampled_image_sz):
#     case_id = pair_name.split("_")
#     case_id =case_id[0]+"_"+case_id[1]
#     slandmark_index_path = os.path.join(dirlab_folder_path,case_id+"_EXP_index.npy")
#     tlandmark_index_path = os.path.join(dirlab_folder_path,case_id+"_INSP_index.npy")
#     slandmark_index = read_index(slandmark_index_path)
#     tlandmark_index = read_index(tlandmark_index_path)
#     slandmark_index = slandmark_index*(resampled_image_sz-1)/(org_image_sz-1)
#     tlandmark_index = tlandmark_index*(resampled_image_sz-1)/(org_image_sz-1)
#     s_spacing, s_origin = get_spacing_and_origin(case_id+"_EXP")
#     t_spacing, t_origin= get_spacing_and_origin(case_id+"_INSP")
#     assert (s_spacing == t_spacing).all()
#     physical_spacing = (org_image_sz-1)*s_spacing/(resampled_image_sz-1) # we assume source and target has the same spacing
#     return slandmark_index, tlandmark_index, physical_spacing, s_origin, t_origin
#
#
# def eval_on_dirlab_per_case(forward_map,inv_map, pair_name,moving, target,record_path):
#     transform_shape = np.array(forward_map.shape[2:])
#     slandmark_index,tlandmark_index, physical_spacing, s_origin, t_origin = get_landmark(pair_name,transform_shape)
#     spacing = 1./(transform_shape-1)
#     slandmark_img_coord = spacing*slandmark_index
#     tlandmark_img_coord = spacing*tlandmark_index
#     slandmark_physical = physical_spacing*slandmark_index+s_origin
#     tlandmark_physical = physical_spacing*tlandmark_index+t_origin
#     # target = target.squeeze().clone()
#     # for coord in tlandmark_index:
#     #     coord_int  = [int(c) for c in coord]
#     #     target[coord_int[0],coord_int[1],coord_int[2]] = 10.
#     # save_3D_img_from_numpy(target.detach().cpu().numpy().squeeze(),"/playpen-raid2/zyshen/debug/{}_debug.nii.gz".format(pair_name))
#
#
#     tlandmark_img_coord_reshape = torch.Tensor(tlandmark_img_coord.transpose(1,0)).view([1,3,-1,1,1])
#     tlandmark_img_coord_reshape = tlandmark_img_coord_reshape.to(forward_map.device)
#     ts_landmark_img_coord = py_utils.compute_warped_image_multiNC(forward_map, tlandmark_img_coord_reshape*2-1, spacing, 1, zero_boundary=False,use_01_input=False)
#     ts_landmark_img_coord = ts_landmark_img_coord.squeeze().transpose(1,0).detach().cpu().numpy()
#     diff_ts = (slandmark_img_coord - ts_landmark_img_coord)/spacing*physical_spacing
#
#
#     # target = target.squeeze().clone()
#     # for coord in ts_landmark_img_coord:
#     #     coord_int  = [int(c) for c in coord/spacing]
#     #     target[coord_int[0],coord_int[1],coord_int[2]] = 10.
#     # save_3D_img_from_numpy(target.detach().cpu().numpy().squeeze(),"/playpen-raid2/zyshen/debug/{}_debug_warped.nii.gz".format(pair_name))
#
#     slandmark_img_coord_reshape = torch.Tensor(slandmark_img_coord.transpose(1, 0)).view([1, 3, -1, 1, 1])
#     slandmark_img_coord_reshape = slandmark_img_coord_reshape.to(inv_map.device)
#     st_landmark_img_coord = py_utils.compute_warped_image_multiNC(inv_map, slandmark_img_coord_reshape * 2 - 1,
#                                                                   spacing, 1, zero_boundary=False, use_01_input=False)
#     st_landmark_img_coord = st_landmark_img_coord.squeeze().transpose(1, 0).detach().cpu().numpy()
#     landmark_saving_folder = os.path.join(record_path,"landmarks")
#     os.makedirs(landmark_saving_folder, exist_ok=True)
#     save_vtk(os.path.join(landmark_saving_folder,"{}_source.vtk".format(pair_name)),{"points":transfer_into_itk_coord(slandmark_physical)})
#     save_vtk(os.path.join(landmark_saving_folder,"{}_target.vtk".format(pair_name)),{"points":transfer_into_itk_coord(tlandmark_physical)})
#     save_vtk(os.path.join(landmark_saving_folder,"{}_target_warp_to_source.vtk".format(pair_name)),{"points":transfer_into_itk_coord(ts_landmark_img_coord / spacing*physical_spacing+s_origin)})
#     save_vtk(os.path.join(landmark_saving_folder,"{}_source_warp_to_target.vtk".format(pair_name)),{"points":transfer_into_itk_coord(st_landmark_img_coord / spacing*physical_spacing+t_origin)})
#     diff_st = (tlandmark_img_coord - st_landmark_img_coord) / spacing * physical_spacing
#
#     return np.linalg.norm(diff_ts,ord=2,axis=1).mean(), np.linalg.norm(diff_st,ord=2,axis=1).mean()
#





def evaluate_on_dirlab(inv_map,dirlab_id,moving_itk, target_itk,record_path):
    MAPPING = {
         "copd_000006" : "12042G" ,
         "copd_000007" : "12105E" ,
         "copd_000008" : "12109M" ,
         "copd_000009" : "12239Z" ,
         "copd_000010" : "12829U" ,
         "copd_000001" : "13216S" ,
         "copd_000002" : "13528L" ,
         "copd_000003" : "13671Q" ,
         "copd_000004" : "13998W" ,
         "copd_000005" : "17441T"
    }

    COPD_ID = [
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



    def get_dirlab_landmark(case_id):
        # assert case_id in COPD_ID
        exp_landmark_path = os.path.join("./lung_reg/landmarks", MAPPING[case_id] + "_EXP_STD_USD_COPD.vtk")
        insp_landmark_path = os.path.join("./lung_reg/landmarks", MAPPING[case_id] + "_INSP_STD_USD_COPD.vtk")
        exp_landmark = read_vtk(exp_landmark_path)["points"]
        insp_landmark = read_vtk(insp_landmark_path)["points"]
        return exp_landmark, insp_landmark

    def read_vtk(path):
        data = pv.read(path)
        data_dict = {}
        data_dict["points"] = data.points.astype(np.float32)
        data_dict["faces"] = data.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        for name in data.array_names:
            try:
                data_dict[name] = data[name]
            except:
                pass
        return data_dict

    def warp_points(points, inv_map, case_id):
        """
        in easyreg the inv transform coord is from [0,1], so here we need to read mesh in voxel coord and then normalized it to [0,1],
        the last step is to transform warped mesh into word/ voxel coord
        the transformation map use default [0,1] coord unless the ref img is provided
        here the transform map is  in inversed voxel space or in  inversed physical space ( width,height, depth)
        but the points should be in standard voxel/itk space (depth, height, width)
        :return:
        """

        import numpy as np
        import torch.nn.functional as F
        # first make everything in voxel coordinate, depth, height, width
        phi_sz = np.array(inv_map.shape[1:])
        moving_img = moving_itk
        moving_spacing = moving_img.GetSpacing()
        moving_spacing = np.array(moving_spacing)
        moving_origin = moving_img.GetOrigin()
        moving_origin = np.array(moving_origin)
        img_sz = moving_img.GetSize()
        standard_spacing = 1 / (np.array(img_sz) - 1)  # depth, height, width

        target_img = target_itk
        target_spacing = target_img.GetSpacing()
        target_spacing = np.array(target_spacing)
        target_origin = target_img.GetOrigin()
        target_origin = np.array(target_origin)

        # moving_spacing = np.array(COPD_info[case_id]["exp"]["spacing"])
        # moving_origin =  np.array(COPD_info[case_id]["exp"]["origin"])
        #
        # target_spacing = np.array(COPD_info[case_id]["insp"]["spacing"])
        # target_origin = np.array(COPD_info[case_id]["insp"]["origin"])

        # moving = sitk.GetArrayFromImage(moving_img)
        # slandmark_index = (points-moving_origin) / moving_spacing
        # for coord in slandmark_index:
        #     coord_int  = [int(c) for c in coord]
        #     moving[coord_int[2],coord_int[1],coord_int[0]] = 2.
        # save_3D_img_from_numpy(moving,"/playpen-raid2/zyshen/debug/{}_padded.nii.gz".format(dirlab_id+"_moving"),
        #                        spacing=moving_img.GetSpacing(), orgin=moving_img.GetOrigin(), direction=moving_img.GetDirection())

        points = (points - moving_origin) / moving_spacing * standard_spacing
        points = points * 2 - 1
        grid_sz = [1] + [points.shape[0]] + [1, 1, 3]  # 1*N*1*1*3
        grid = points.reshape(*grid_sz)
        grid = torch.Tensor(grid).cuda()
        inv_map_sz = [1, 3] + list(phi_sz)  # width,height, depth
        inv_map = inv_map.view(*inv_map_sz)  # 1*3*X*Y*Z
        points_wraped = F.grid_sample(inv_map, grid, mode='bilinear', padding_mode='border',
                                      align_corners=True)  # 1*3*N*1*1
        points_wraped = points_wraped.detach().cpu().numpy()
        points_wraped = np.transpose(np.squeeze(points_wraped))
        points_wraped = np.flip(points_wraped, 1) / standard_spacing * target_spacing + target_origin

        warp = sitk.GetArrayFromImage(target_img)
        wlandmark_index = (points_wraped - target_origin) / target_spacing
        for coord in wlandmark_index:
            coord_int = [int(c) for c in coord]
            warp[coord_int[2], coord_int[1], coord_int[0]] = 2.
        save_3D_img_from_numpy(warp, "/playpen-raid2/zyshen/debug/{}_debug.nii.gz".format(dirlab_id+"warp"))

        return points_wraped


    assert dirlab_id in COPD_ID, "{} doesn't belong to ten dirlab cases:{}".format(dirlab_id, COPD_ID.keys())
    exp_landmark, insp_landmark = get_dirlab_landmark(dirlab_id)
    warped_landmark = warp_points(exp_landmark, inv_map,dirlab_id)
    diff = warped_landmark - insp_landmark
    diff_norm = np.linalg.norm(diff, ord=2, axis=1)
    print("before register landmark error norm: {}".format(
        np.linalg.norm(exp_landmark - insp_landmark, ord=2, axis=1).mean()))
    print("after register landmark error norm: {}".format(diff_norm.mean()))
    return diff_norm.mean()



def eval_on_dirlab(forward_map,inverse_map, pair_name_list,pair_path_list, moving, target, record_path=None):
    diff_ts, diff_st = [], []
    for _forward_map, _inv_map, pair_name,s_pth, t_pth in zip(forward_map,inverse_map,pair_name_list,pair_path_list[0],pair_path_list[1]):
        source_sitk = sitk.ReadImage(s_pth)
        target_sitk = sitk.ReadImage(t_pth)
        if pair_name in COPD_ID:
            _diff_st = evaluate_on_dirlab(_inv_map,pair_name,source_sitk,target_sitk, record_path)
            _diff_ts = evaluate_on_dirlab(_forward_map,pair_name,target_sitk,source_sitk, record_path)
            diff_ts.append(_diff_ts), diff_st.append(_diff_st)
    print("dirlab landmark source to target error:{}:{}".format(pair_name_list,diff_st))
    print("dirlab landmark target_to source error:{}:{}".format(pair_name_list,diff_ts))
