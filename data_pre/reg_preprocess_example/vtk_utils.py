import vtk
import pyvista as pv
import numpy as np


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


def convert_faces_into_file_format(faces):
    ind = np.ones([faces.shape[0], 1]) * 3
    faces = np.concatenate((ind, faces), 1).astype(np.int64)
    return faces.flatten()


def save_vtk(fpath, attr_dict):
    points = attr_dict["points"]
    faces = attr_dict["faces"] if "faces" in attr_dict else None
    if faces is not None:
        faces = convert_faces_into_file_format(faces)
        data = pv.PolyData(points, faces)
    else:
        data = pv.PolyData(points)
    for key, item in attr_dict.items():
        if key not in ["points", "faces"]:
            if len(points) == len(item):
                data.point_arrays[key] = item
    data.save(fpath)


if __name__ == "__main__":
    # file_path = "/playpen-raid1/Data/UNC_vesselParticles/case1_exp.vtk"
    # data_dict = read_vtk(file_path)
    # for key, val in data_dict.items():
    #     print("attri {} with size {}".format(key, val.shape))
    file_path = "/playpen-raid1/zyshen/debug/body_registration/source.ply"
    data_dict = read_vtk(file_path)
    for key, val in data_dict.items():
        print("attri {} with size {}".format(key, val.shape))
    file_path = "/playpen-raid1/zyshen/debug/body_registration/source_debug.vtk"
    save_vtk(file_path, data_dict)
