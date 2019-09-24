import os
import subprocess
from data_pre.reg_data_utils import  read_txt_into_list
txt_path = '/playpen/zyshen/debugs/get_val_and_debug_res/example_for_demo.txt'
path_list = read_txt_into_list(txt_path)
target_path = '/playpen/zyshen/debugs/examples'
os.makedirs(target_path,exist_ok=True)
file_list =[]
for fl in path_list:
    file_list += fl
cmd=''
for f in file_list:
    cmd += '\n cp ' + f + ' ' + os.path.join(target_path,os.path.split(f)[1])
process = subprocess.Popen(cmd, shell=True)
process.wait()

