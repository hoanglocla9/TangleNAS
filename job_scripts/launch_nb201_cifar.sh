#!/bin/bash
seeds=(9001 9002 9003 9004)
optimizers=("darts_v1" "darts_v2" "gdas" "drnas")
datasets=("cifar10" "cifar100")
for dataset in "${datasets[@]}"
do
    for optimizer in "${optimizers[@]}"
    do
        for seed in "${seeds[@]}"
        do
		sbatch job_scripts/job_nb201_cifar.sh $dataset $optimizer "nb201" $seed
        done
    done
done
