import os
import numpy as np
from glob import glob
import SimpleITK as sitk
import pyvista as pv
import json
from data_pre.reg_preprocess_example.img_sampler import DataProcessing

case_sampled_info = {}


COPD_ID={
    "copd1":  "copd_000001",
    "copd2":  "copd_000002",
    "copd3":  "copd_000003",
    "copd4":  "copd_000004",
    "copd5":  "copd_000005",
    "copd6":  "copd_000006",
    "copd7":  "copd_000007",
    "copd8":  "copd_000008",
    "copd9":  "copd_000009",
    "copd10": "copd_000010"
}

ID_COPD={
    "12042G":"copd6",
    "12105E":"copd7",
    "12109M":"copd8",
    "12239Z":"copd9",
    "12829U":"copd10",
    "13216S":"copd1",
    "13528L":"copd2",
    "13671Q":"copd3",
    "13998W":"copd4",
    "17441T":"copd5"
}


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

def load_json(file_path):
    import json
    with open(file_path) as f:
        data_dict = json.load(f)
    return data_dict

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def normalize_intensity(img, linear_clip=False):
    """
    a numpy image, normalize into intensity [-1,1]
    (img-img.min())/(img.max() - img.min())
    :param img: image
    :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
    :return:
    """
    if linear_clip:
        img = img - img.min()
        normalized_img = img / np.percentile(img, 95) * 0.95
    else:
        min_intensity = img.min()
        max_intensity = img.max()
        normalized_img = (img - img.min()) / (max_intensity - min_intensity)
    return normalized_img



def process_image(img, is_label=False):
    """
    :param img: numpy image
    :return:
    """
    if not is_label:
        img[img<-1000] = -1000
        img[img>-200] = -200
        # img = normalize_intensity(img)
    else:
        img[img > 400] = 0
        img[img != 0] = 1
    return img



def process_high_to_raul_format(image_path,label_path,case_id, saving_folder=None,is_insp=True):
    img_sitk = sitk.ReadImage(image_path)
    label_sitk = sitk.ReadImage(label_path)
    processed_img = DataProcessing.resample_image_itk_by_spacing_and_size(img_sitk, output_spacing = np.array([1.,1.,1.]), output_size=[350,350,350], output_type=None,
                                               interpolator=sitk.sitkBSpline, padding_value=-1024, center_padding=True)

    mode = "INSP" if is_insp else "EXP"
    saving_path = os.path.join(saving_folder,COPD_ID[case_id]+"_{}_img.nii.gz").format(mode)
    img_np = sitk.GetArrayFromImage(processed_img)
    img_np = process_image(img_np.astype(np.float32))
    sitk_img = sitk.GetImageFromArray(img_np)
    sitk_img.SetOrigin(processed_img.GetOrigin())
    sitk_img.SetSpacing(processed_img.GetSpacing())
    sitk_img.SetDirection(processed_img.GetDirection())
    sitk.WriteImage(sitk_img,saving_path)
    processed_label = DataProcessing.resample_image_itk_by_spacing_and_size(label_sitk,
                                                                          output_spacing=np.array([1., 1., 1.]),
                                                                          output_size=[350, 350, 350], output_type=None,
                                                                          interpolator=sitk.sitkNearestNeighbor,
                                                                          padding_value=0, center_padding=True)
    saving_path = os.path.join(saving_folder, COPD_ID[case_id] + "_{}_label.nii.gz").format(mode)
    label_np = sitk.GetArrayFromImage(processed_label)
    label_np = process_image(label_np.astype(np.float32),is_label=True)
    sitk_label = sitk.GetImageFromArray(label_np)
    sitk_label.SetOrigin(processed_label.GetOrigin())
    sitk_label.SetSpacing(processed_label.GetSpacing())
    sitk_label.SetDirection(processed_label.GetDirection())
    sitk.WriteImage(sitk_label, saving_path)
    case_sampled_info[case_id+"_"+mode] = {"spacing":processed_img.GetSpacing(), "origin":processed_img.GetOrigin(),"path":saving_path}

def file_exist(path_list):
    for path in path_list:
        if not os.path.isfile(path):
            print(path)

high_folder_path = "/playpen-raid1/lin.tian/data/raw/DIRLABCasesHighRes"
img_insp_key = "*/*_INSP.nrrd"
saving_folder = "/playpen-raid2/zyshen/data/lung_resample_350_lin"
os.makedirs(saving_folder,exist_ok=True)
img_insp_path_list= glob(os.path.join(high_folder_path,img_insp_key))
img_exp_path_list = [path.replace("INSP","EXP") for path in img_insp_path_list]
img_insp_label_path_list=  [path.replace("INSP.nrrd","INSP_label.nrrd") for path in img_insp_path_list]
img_exp_label_path_list = [path.replace("EXP.nrrd","EXP_label.nrrd") for path in img_exp_path_list]
case_id_list = [os.path.split(path)[-1].split("_")[0] for path in img_insp_path_list]
case_sampled_json_saving_path = os.path.join(saving_folder,"dirlab_350_sampled.json")
file_exist(img_insp_path_list)
file_exist(img_exp_path_list)
file_exist(img_insp_label_path_list)
file_exist(img_exp_label_path_list)

for insp_path,insp_label_path, case_id in zip(img_insp_path_list,img_insp_label_path_list,case_id_list):
    process_high_to_raul_format(insp_path,insp_label_path,case_id,saving_folder,is_insp=True)
for exp_path,exp_label_path, case_id in zip(img_exp_path_list,img_exp_label_path_list, case_id_list):
    try:
        process_high_to_raul_format(exp_path,exp_label_path, case_id,saving_folder, is_insp=False)
    except:
        print(exp_path)
print(case_sampled_info)
save_json(case_sampled_json_saving_path,case_sampled_info)