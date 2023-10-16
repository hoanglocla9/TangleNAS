#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J nb201 # sets the job name. If not specified, the file name will be used as job name
dataset=$1
optimizer=$2
seed=$3
python -m search.experiment_progressive_search --dataset ${dataset}  --optimizer ${optimizer}  --searchspace darts --seed ${seed} --submission
