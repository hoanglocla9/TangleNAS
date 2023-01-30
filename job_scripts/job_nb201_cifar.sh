#!/bin/bash
#SBATCH -p partition_name
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J nb201 # sets the job name. If not specified, the file name will be used as job name
dataset=$1
optimizer=$2
search_space=$3
seed=$4
python -m search.experiment_search  --dataset ${dataset}  --optimizer ${optimizer}  --searchspace ${search_space} --seed ${seed} --path_to_benchmark NAS-Bench-201-v1_0-e61699.pth 

