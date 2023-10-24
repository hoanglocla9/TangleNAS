#!/bin/bash
optimizers=("drnas" "darts_v1" "gdas")
train_portions=(0.5 0.8)
for optimizer in "${optimizers[@]}"
do
    for train_portion in "${train_portions[@]}"
    do
        exp_name="charlm-${optimizer}-${train_portion}"
        echo Submitting job $exp_name
        sbatch -J $exp_name job_scripts/job_charlm.sh $optimizer $train_portion
    done
done