#!/bin/bash
seeds=(9001 9002 9003 9004)
optimizers=("drnas")
datasets=("cifar10")
for dataset in "${datasets[@]}"
do
    for optimizer in "${optimizers[@]}"
    do
        for seed in "${seeds[@]}"
        do
		sbatch job_scripts/job_darts_drnas.sh $dataset $optimizer $seed
        done
    done
done