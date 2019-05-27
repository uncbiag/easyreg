#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-0
#SBATCH --mem=6G
#SBATCH --output=oai_iter_2.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python single_pair_registration.py  --gpu=0 --llf=True --task_name="rdmm_iter_wkw_formul_025_1_omt_0001tv_2step_200sym_minstd_005_allinterp_maskv_epdffix" --mermaid_net_json_pth="mermaid_settings/cur_settings_adpt_lddmm_new_fix_wkw_4.json"