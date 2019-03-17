#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --output=nifty_reg_affine.txt
#SBATCH -t 2-

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_oasis.py  --gpu=-1 --llf=True