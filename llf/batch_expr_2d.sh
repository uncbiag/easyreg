#!/bin/sh

folder=./batch_folder_2d/expr_2d_
for i in {1..8}
do
    path="$folder$i.sh"
    sbatch  $path
done