#!/bin/bash
seeds=(9003)
optimizers=("darts_v2")
datasets=("cifar10")
for dataset in "${datasets[@]}"
do
    for optimizer in "${optimizers[@]}"
    do
        for seed in "${seeds[@]}"
        do
		sbatch job_scripts/job_darts.sh $dataset $optimizer $seed
        done
    done
done
