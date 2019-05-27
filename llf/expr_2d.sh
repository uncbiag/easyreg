#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=16g
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --output=rddmm_maxstd008_omt_01_100iter.txt
#SBATCH -t 5-

source activate torch
cd /pine/scr/z/y/zyshen/reg_clean/demo/
srun python demo_for_2d_adap_reg.py --llf=True