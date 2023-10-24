#!/bin/bash

seeds=(9016)
models=("re")
master_port=20001


for model in "${models[@]}"
    do
    for seed in "${seeds[@]}"
    do
        sbatch ./job_scripts/train_nanogpt_${model}_best.sh $seed $master_port
        master_port=$((master_port + 1))
    done
done
