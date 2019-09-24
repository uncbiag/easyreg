#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5-0
#SBATCH --mem=48G
#SBATCH --output=oai_learn_0_1.txt
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/ #/pine/scr/z/h/zhiding/zyshen/reg_clean/demo/
srun python demo_for_reg.py  --gpu=0 --llf=True --task_name="rdmm_learn_sm004_sym_400_thisis_std005" --mermaid_net_json_pth="mermaid_settings/cur_settings_adpt_lddmm_for_oai_opt4.json"