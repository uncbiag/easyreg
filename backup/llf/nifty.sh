#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=30g
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --output=ants_baseline_200_300.txt
#SBATCH -t 5-

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_oasis.py  --gpu=-1 --llf=True