#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --output=expr_2d_0.txt
#SBATCH -t 5-

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
set_path="/pine/scr/z/y/zyshen/reg_clean/llf/batch_folder_2d/"

srun python demo_for_2d_adap_reg_batch_mutli.py --llf=True  --expr_name="lddmm"