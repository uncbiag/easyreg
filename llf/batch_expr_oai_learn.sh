#!/bin/sh

folder=./batch_folder_oai_learn/expr_oai_
for i in {1..5}
do
    path="$folder$i.sh"
    sbatch  $path
done