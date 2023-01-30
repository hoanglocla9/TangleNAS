#!/bin/bash
seeds=(9001 9002 9003 9004)
optimizers=("darts_v1" "darts_v2" "gdas" "drnas" "spos")
datasets=("imgnet16_120")
for dataset in "${datasets[@]}"
do
    for optimizer in "${optimizers[@]}"
    do
        for seed in "${seeds[@]}"
        do
		sbatch job_scripts/job_nats_imgnet.sh $dataset $optimizer "nats_v1" $seed
        done
    done
done