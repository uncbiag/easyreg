#!/bin/sh

folder=./batch_folder_oai_iter/expr_oai_
for i in {1..7}
do
    path="$folder$i.sh"
    sbatch  $path
done