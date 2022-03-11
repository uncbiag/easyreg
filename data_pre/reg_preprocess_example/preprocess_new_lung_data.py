import h5py
import SimpleITK as sitk
import os, sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('../..'))
sys.path.insert(0,os.path.abspath('../../easyreg'))
import numpy as np
import glob
from  easyreg.reg_data_utils import write_list_into_txt
from multiprocessing import Process
h5_path = "/playpen-raid1/Data/UNC_Registration.h5"
output_path = "/playpen-raid2/Data/Lung_Registration_clamp_normal_transposed"
os.makedirs(output_path,exist_ok=True)
#['Expiration_CT', 'Expiration_CT.key', 'Expiration_CT.missing',
# 'Expiration_CT.origin', 'Expiration_CT.spacing', 'Expiration_labelmap',
# 'Expiration_labelmap.key', 'Expiration_labelmap.missing', 'Expiration_labelmap.origin',
# 'Expiration_labelmap.spacing', 'Inspiration_CT', 'Inspiration_CT.key',
# 'Inspiration_CT.missing', 'Inspiration_CT.origin', 'Inspiration_CT.spacing',
# 'Inspiration_labelmap', 'Inspiration_labelmap.key', 'Inspiration_labelmap.missing',
# 'Inspiration_labelmap.origin', 'Inspiration_labelmap.spacing', 'Inspiration_local_histogram_lm',
# 'Inspiration_local_histogram_lm.key', 'Inspiration_local_histogram_lm.missing',
# 'Inspiration_local_histogram_lm.origin', 'Inspiration_local_histogram_lm.spacing']

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



def process_image(img, fname, is_label=False):
    """
    :param img: numpy image
    :return:
    """
    if not is_label:
        img[img < -1000] = -1000
        img[img > -200] = -200
        # img = normalize_intensity(img)
    else:
        img[img >400] =0
        img[img != 0] = 1
        # img[img==2]=1
        # img[img==3]=2
        # assert list(np.unique(img))==[0,1,2],"the fname {} has label {} with label 3 density{}".format(fname,list(np.unique(img)),np.sum(img==3))

    return img



def process_lung_data(index_list):
    f = h5py.File(h5_path, 'r')
    modaility = ['Expiration_CT','Expiration_labelmap','Inspiration_CT','Inspiration_labelmap','Inspiration_local_histogram_lm']
    mod_suffix = ['_img','_label','_img','_label','_hist']
    is_label = [False, True, False, True, None]

    for ind,mod in enumerate(modaility):
        atr_key = mod+'.key'
        atr_origin = mod + '.origin'
        atr_spacing = mod + '.spacing'
        for i in index_list:
            img = f[mod][i]
            img = process_image(img,f[atr_key][i][1],is_label[ind]) if is_label[ind] is not None else img
            folder_name = f[atr_key][i][0]
            fname = f[atr_key][i][1]
            origin = f[atr_origin][i].astype(np.float64)
            spacing = f[atr_spacing][i].astype(np.float64)
            img = np.transpose(img, (2, 1, 0))
            origin = np.flipud(origin)
            spacing = np.flipud(spacing)
            sitk_img = sitk.GetImageFromArray(img)
            sitk_img.SetOrigin(origin)
            sitk_img.SetSpacing(spacing)
            output_folder = os.path.join(output_path,folder_name)
            os.makedirs(output_folder,exist_ok=True)
            sitk.WriteImage(sitk_img,os.path.join(output_folder, fname+mod_suffix[ind]+".nii.gz"))


def get_input_file(refer_folder, output_txt):
    source_image_path_list = glob.glob(os.path.join(refer_folder,"**","*EXP*img*"))
    source_label_path_list = [path.replace("_img.nii.gz","_label.nii.gz") for path in source_image_path_list]
    target_image_path_list = [path.replace("_EXP_","_INSP_") for path in source_image_path_list]
    target_label_path_list = [path.replace("_img.nii.gz","_label.nii.gz") for path in target_image_path_list]
    num_file = len(source_image_path_list)
    file_list = [[source_image_path_list[i], target_image_path_list[i], source_label_path_list[i], target_label_path_list[i]] for i in
                 range(num_file)]
    write_list_into_txt(output_txt,file_list)



num_of_workers=20

split_index = np.array_split(np.array(range(999)), num_of_workers)



procs = []
for i in range(num_of_workers):
    p = Process(target=process_lung_data, args=(split_index[i],))
    p.start()
    print("pid:{} start:".format(p.pid))
    procs.append(p)

for p in procs:
    p.join()

#
# txt_output_path = "/playpen-raid2/zyshen/data/lung_new_reg"
# os.makedirs(txt_output_path,exist_ok=True)
# output_txt = os.path.join(txt_output_path,"pair_path_list.txt")
# get_input_file(txt_output_path,output_txt)






