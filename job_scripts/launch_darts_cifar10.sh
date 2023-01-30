#!/bin/bash
seeds=(9001 9002 9003 9004)
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
