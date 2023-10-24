#!/bin/bash

seeds=(9001 9006 9011 9016)
master_port=19001

for seed in "${seeds[@]}"
do
    sbatch ./job_scripts/train_nanogpt_drnas_best.sh $seed $master_port
    master_port=$((master_port + 1))
done
