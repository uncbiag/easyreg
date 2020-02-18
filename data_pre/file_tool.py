from easyreg.reg_data_utils import write_list_into_txt,read_txt_into_list
import numpy as np
import os
from glob import glob

def split_txt(input_txt,num_split, output_folder):
    pairs = read_txt_into_list(input_txt)
    output_splits = np.split(pairs, num_split)
    output_splits = list(output_splits)
    for i in num_split:
        write_list_into_txt(os.path.join(output_folder, 'p{}.txt'.format(i)),output_splits[i])


def get_txt_file(path, ftype, output_txt):
    f_pth = os.path.join(path,"**",ftype)
    file_list = glob(f_pth,recursive=True)
    file_list = [[f] for f in file_list]
    write_list_into_txt(output_txt, file_list)


# num_split= 20
# input_txt = ''

path = "/playpen-raid/zyshen/data/oai_reg/test_aug/reg/res/records/original_sz"
ftype= "*_warped.nii.gz"
output_txt ="/playpen-raid/zyshen/data/oai_reg/test_aug/file_path_list.txt"
get_txt_file(path,ftype,output_txt)

path = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records/original_sz"
ftype= "*_warped.nii.gz"
output_txt ="/playpen-raid/zyshen/data/lpba_reg/test_aug/file_path_list.txt"
get_txt_file(path,ftype,output_txt)
