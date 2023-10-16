#!/bin/bash

optimizers=("darts" "gdas" "drnas")
datasets=("CIFAR10" "CIFAR100")

for dataset in "${datasets[@]}"
do
    for optimizer in "${optimizers[@]}"
    do
        expname="${optimizer}-${dataset}-5050-wev2-finetune"
        sbatch -J $expname job_scripts/job_autoformer_wev2_5050_finetune.sh $optimizer $dataset
        ls job_scripts/job_autoformer_wev2_5050_finetune.sh
        echo "lalalala"
    done
done