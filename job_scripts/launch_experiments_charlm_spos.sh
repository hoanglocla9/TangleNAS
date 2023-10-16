#!/bin/bash
optimizer="spos"
train_portions=(0.5 0.8)
for train_portion in "${train_portions[@]}"
do
    exp_name="charlm-${optimizer}-${train_portion}"
    echo Submitting job $exp_name
    sbatch -J $exp_name job_scripts/job_charlm_spos.sh $optimizer $train_portion
done