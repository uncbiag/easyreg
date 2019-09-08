#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=6g
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --output=expr_2d_10.txt
#SBATCH -t 5-

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
set_path="/pine/scr/z/y/zyshen/reg_clean/llf/batch_folder_2d/"

srun python demo_for_2d_adap_reg.py --llf=True  --expr_name="rddmm_maxstd008_omt_1_100iter" --mermaid_setting_path=$set_path"s_10.json"