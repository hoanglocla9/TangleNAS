#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 #rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -a 9001-9004 # job array index

optimizer=$1
train_portion=$2
seed=$SLURM_ARRAY_TASK_ID
if [ $optimizer -eq "spos" ]; then
    max_iters=20000
else
    max_iters=10000
fi
echo "optimizer: ${optimizer}, seed: ${seed}, max_iters: ${max_iters}"

python_cmd="python toy_search_spaces/nanoGPT/train_spos.py --config toy_search_spaces/nanoGPT/config/train_shakespeare_char.py --optimizer ${optimizer} --seed ${seed} --max_iters ${max_iters} --train_portion ${train_portion}"

eval $python_cmd