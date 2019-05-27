#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-0
#SBATCH --mem=20G
#SBATCH --output=reg_adpt_lddamm_wkw_formul_05_1_omt_2step_004_008_sm_004maskv_10sigaff.txt
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_lung_given_regularizer_registration.py  --gpu=0 --llf=True --task_name="reg_adpt_lddamm_wkw_formul_05_1_omt_2step_004_008_sm_004maskv_10sigaff_with_affine_init" --mermaid_net_json_pth="mermaid_settings/cur_settings_adpt_lddmm_for_lung_opt.json"