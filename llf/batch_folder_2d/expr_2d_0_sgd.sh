#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-0
#SBATCH --mem=20G
#SBATCH --output=sgd_lddmm_smoother_map.txt
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
set_path="/pine/scr/z/y/zyshen/reg_clean/llf/batch_folder_2d/"

srun python demo_for_2d_adap_reg_batch.py --llf=True  --expr_name="sgd_lddmm_smoother_map"