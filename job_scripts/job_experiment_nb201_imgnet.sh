#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -a 9001-9004 # job array index

dataset=$1
optimizer=$2
search_space=$3
use_we=$4
use_we_v2=$5
submission=$6
train_portion=$7
seed=$SLURM_ARRAY_TASK_ID


if [ "$use_we" == "we" ]; then
    we_flag=""
else
    we_flag="--no_weight_entanglement"
fi

if [ $submission -eq 1 ]; then
    submission_flag="--submission"
else
    submission_flag=""
fi

if [ $use_we_v2 -eq 1 ]; then
    we_v2_flag="--use_we_v2"
else
    we_v2_flag=""
fi

python_cmd="python -m search.experiment_search --dataset ${dataset} --optimizer ${optimizer} --searchspace ${search_space} --seed ${seed} ${submission_flag} ${we_flag} ${we_v2_flag} --path_to_benchmark /path/to/NAS-Bench-201-v1_0-e61699.pth --train_portion ${train_portion} --data_path /path/to/ImageNet16"

eval $python_cmd
