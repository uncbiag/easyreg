#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5-0
#SBATCH --mem=16G
#SBATCH --output=rdmm_iter_omt025_locals_006_clampx_05_accumiter_softmaxw_maskmpow2_b2_fixed_cp.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python single_pair_registration.py  --gpu=0 --llf=True