import os


def print_txt(txt, output_path):
    with open(output_path,"w") as f:
        f.write(txt)






key_w_list = [10,20,30,40,60,80,100]
key_w_list2= ["1d","atlas",'rand','bspline','aug']


for key_w2 in key_w_list2:
    output_path = '/playpen-raid/zyshen/debug/llf_output/par/oai_seg_{}'.format(key_w2)
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
srun python demo_for_seg_train.py -o /pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever/gen_lresol_{} -dtn={}case -tn=seg_par -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/oai_seg_par -g=0
srun python demo_for_seg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/oai_seg_par -txt=/pine/scr/z/y/zyshen/data/oai_seg/baseline/10case/test/file_path_list.txt  -m=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever/gen_lresol_{}/{}case/seg_par/checkpoints/model_best.pth.tar  -o=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_{}/{}case -g=0
""".format(key_w,key_w2,key_w2,key_w,key_w2,key_w,key_w2,key_w)

        print_txt(txt,os.path.join(output_path,file_name_list[i]))


#
# key_w_list = [10,20,30,40,60,80,100]
# key_w_list2= ["1d"]
#
#
# for key_w2 in key_w_list2:
#     output_path = '/playpen-raid/zyshen/debug/llf_output/oai_seg_{}'.format(key_w2)
#     os.makedirs(output_path, exist_ok=True)
#     file_name_list = ["oai_expr_{}.sh".format(i) for i in range(1,len(key_w_list)+1)]
#
#     for i, key_w in enumerate(key_w_list):
#         txt = """#!/bin/bash
# #SBATCH -p general
# #SBATCH -N 1
# #SBATCH --mem=8g
# #SBATCH -n 1
# #SBATCH -c 6
# #SBATCH --output=oai_expr_aug_1d_{}.txt
# #SBATCH -t 3-
#
#
# source activate torch4
# cd /pine/scr/z/y/zyshen/reg_clean/mermaid/mermaid_demos
# srun python /pine/scr/z/y/zyshen/reg_clean/mermaid/mermaid_demos/gen_aug_samples.py --txt_path=/pine/scr/z/y/zyshen/data/oai_reg/train_with_{}/momentum_lresol.txt  --mermaid_setting_path=/pine/scr/z/y/zyshen/reg_clean/debug/settings/oai_reg/mermaid_nonp_settings.json --output_path=/pine/scr/z/y/zyshen/data/oai_seg/baseline/aug/gen_lresol_1d/{}case
# """.format(key_w,key_w,key_w,key_w)
#
#         print_txt(txt,os.path.join(output_path,file_name_list[i]))


# key_w_list = [5,10,15,20,25]
# key_w_list2= ["1d"]
#
#
# for key_w2 in key_w_list2:
#     output_path = '/playpen-raid/zyshen/debug/llf_output/lpba_seg_{}'.format(key_w2)
#     os.makedirs(output_path, exist_ok=True)
#     file_name_list = ["lpba_expr_{}.sh".format(i) for i in range(1,len(key_w_list)+1)]
#
#     for i, key_w in enumerate(key_w_list):
#         txt = """#!/bin/bash
# #SBATCH -p general
# #SBATCH -N 1
# #SBATCH --mem=8g
# #SBATCH -n 1
# #SBATCH -c 6
# #SBATCH --output=lpba_expr_aug_1d_{}.txt
# #SBATCH -t 3-
#
#
# source activate torch4
# cd /pine/scr/z/y/zyshen/reg_clean/mermaid/mermaid_demos
# srun python /playpen-raid/zyshen/reg_clean/mermaid/mermaid_demos/gen_aug_samples.py --txt_path=/playpen-raid/zyshen/data/lpba_reg/train_with_{}/lpba_ncc_reg1/momentum_lresol.txt  --mermaid_setting_path=/playpen-raid/zyshen/reg_clean/debug/settings/opt_lddmm/mermaid_nonp_settings.json --output_path=/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/gen_lresol_1d/{}case
#
# """.format(key_w,key_w,key_w)
#
#         print_txt(txt,os.path.join(output_path,file_name_list[i]))





# key_w_list = [5,10,15,20,25]
# key_w_list2 = ["1d","atlas"]
#
# for key_w2 in key_w_list2:
#     output_path = '/playpen-raid/zyshen/debug/llf_output/lpba_seg_{}'.format(key_w2)
#     os.makedirs(output_path, exist_ok=True)
#     file_name_list = ["lpba_expr_{}.sh".format(i) for i in range(1, len(key_w_list) + 1)]
#
#     for i, key_w in enumerate(key_w_list):
#         txt = """#!/bin/bash
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=12
# #SBATCH --time=4-0
# #SBATCH --mem=64G
# #SBATCH --output={}case_seg_{}.txt
# #SBATCH --partition=volta-gpu
# #SBATCH --gres=gpu:1
# #SBATCH --qos=gpu_access
#
# source activate torch4
# cd /pine/scr/z/y/zyshen/reg_clean/demo/
# srun python demo_for_seg_train.py -o /pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/sever/gen_lresol_{} -dtn={}case -tn=seg -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/lpba_seg_aug -g=0
# srun python demo_for_seg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/lpba_seg_aug -txt=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/10case/test/file_path_list.txt  -m=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/sever/gen_lresol_{}/{}case/seg/checkpoints/model_best.pth.tar  -o=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_{}/{}case -g=0
# """.format(key_w,key_w2,key_w2,key_w,key_w2,key_w,key_w2,key_w)
#
#         print_txt(txt, os.path.join(output_path, file_name_list[i]))
#
#
# #



#
# key_w_list = [5,10,15,20,25]
# key_w_list2 = ["multi_reg"]
#
# for key_w2 in key_w_list2:
#     output_path = '/playpen-raid/zyshen/debug/llf_output/lpba_reg_{}'.format(key_w2)
#     os.makedirs(output_path, exist_ok=True)
#
#     for i, key_w in enumerate(key_w_list):
#         for j in range(2):
#             file_name = "lpba_expr_{}.sh".format(i*2+j)
#             txt = """#!/bin/bash
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=12
# #SBATCH --time=6-0
# #SBATCH --mem=24G
# #SBATCH --output={}case_seg_{}.txt
# #SBATCH --partition=volta-gpu
# #SBATCH --gres=gpu:1
# #SBATCH --qos=gpu_access
#
# source activate torch4
# cd /pine/scr/z/y/zyshen/reg_clean/demo/
# python demo_for_easyreg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/opt_lddmm_new_sm -txt=/pine/scr/z/y/zyshen/data/lpba_seg_resize/multi_reg/multi_reg_list_{}/p{}.txt   -o=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/gen_lresol_multi_reg_trans/{}case_p{} -g=0 &
# python demo_for_easyreg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/opt_lddmm_new_sm -txt=/pine/scr/z/y/zyshen/data/lpba_seg_resize/multi_reg/multi_reg_list_{}/p{}.txt   -o=/pine/scr/z/y/zyshen/data/lpba_seg_resize/baseline/aug/gen_lresol_multi_reg_trans/{}case_p{} -g=0
# """.format(key_w,key_w2,key_w,j*2,key_w,j*2,key_w,j*2+1,key_w,j*2+1)
#
#             print_txt(txt, os.path.join(output_path, file_name))
#
# #
# #
# #
# key_w_list = [0,1,2,3,4]
# key_w_list2 = ["test_aug_opt"]
#
# for key_w2 in key_w_list2:
#     output_path = '/playpen-raid/zyshen/debug/llf_output/oai_reg_{}'.format(key_w2)
#     os.makedirs(output_path, exist_ok=True)
#
#     for i, key_w in enumerate(key_w_list):
#             file_name = "oai_expr_{}.sh".format(i)
#             txt = """#!/bin/bash
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=12
# #SBATCH --time=4-0
# #SBATCH --mem=24G
# #SBATCH --output={}case_reg_{}.txt
# #SBATCH --partition=volta-gpu
# #SBATCH --gres=gpu:1
# #SBATCH --qos=gpu_access
#
# source activate torch4
# cd /pine/scr/z/y/zyshen/reg_clean/demo/
# python demo_for_easyreg_eval.py -ts=/pine/scr/z/y/zyshen/reg_clean/debug/settings/opt_lddmm_oai -txt=/pine/scr/z/y/zyshen/data/oai_reg/train_with_test_aug_40/test/opt/p{}.txt   -o=/pine/scr/z/y/zyshen/data/oai_reg/test_aug_opt/p{} -g=0
# """.format(key_w,key_w2,key_w,key_w)
#
#             print_txt(txt, os.path.join(output_path, file_name))
#
#

