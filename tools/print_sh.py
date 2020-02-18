import os


def print_txt(txt, output_path):
    with open(output_path,"w") as f:
        f.write(txt)






key_w_list = [10,20,30,40,60,80,100]
key_w_list2= ["bspline","rand"]


for key_w2 in key_w_list2:
    output_path = '/playpen-raid/zyshen/debug/llf_output/oai_seg_{}'.format(key_w2)
    os.makedirs(output_path, exist_ok=True)
    file_name_list = ["oai_expr_{}.sh".format(i) for i in range(1,len(key_w_list)+1)]

    for i, key_w in enumerate(key_w_list):
        txt = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-0
#SBATCH --mem=96G
#SBATCH --output={}case_seg_{}.txt
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access


source activate torch4
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_seg_train.py -o /pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever/gen_lresol_{} -dtn={}case -tn=seg -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/oai_seg_aug -g=0
srun python demo_for_seg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/oai_seg_aug -txt=/pine/scr/z/y/zyshen/data/oai_seg/baseline/10case/test/file_path_list.txt  -m=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever/gen_lresol_{}/{}case/seg/checkpoints/model_best.pth.tar  -o=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_{}/{}case -g=0
""".format(key_w,key_w2,key_w2,key_w,key_w2,key_w,key_w2,key_w)

        print_txt(txt,os.path.join(output_path,file_name_list[i]))

key_w_list = [5,10,15,20,25]
key_w_list2 = ["bspline", "rand"]

for key_w2 in key_w_list2:
    output_path = '/playpen-raid/zyshen/debug/llf_output/lpba_seg_{}'.format(key_w2)
    os.makedirs(output_path, exist_ok=True)
    file_name_list = ["lpba_expr_{}.sh".format(i) for i in range(1, len(key_w_list) + 1)]

    for i, key_w in enumerate(key_w_list):
        txt = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-0
#SBATCH --mem=64G
#SBATCH --output={}case_seg_{}.txt
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch4
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_seg_train.py -o /pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/sever/gen_lresol_{} -dtn={}case -tn=seg -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/lpba_seg_aug -g=0
srun python demo_for_seg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/lpba_seg_aug -txt=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/10case/test/file_path_list.txt  -m=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/sever/gen_lresol_{}/{}case/seg/checkpoints/model_best.pth.tar  -o=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_{}/{}case -g=0
""".format(key_w,key_w2,key_w2,key_w,key_w2,key_w,key_w2,key_w)

        print_txt(txt, os.path.join(output_path, file_name_list[i]))



