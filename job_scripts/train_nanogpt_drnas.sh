#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J BabyGPT-768-TinyStories-DrNAS-WE # sets the job name. If not specified, the file name will be used as job name

source ~/.bash_profile
conda activate tanglenas

export PYTHONPATH=$(pwd)

# Set default values for command line arguments
seed=9001
train_portion=0.5
master_port=9001

# Check if command line arguments are provided, otherwise use default values
seed="${1:-$seed}"
train_portion="${2:-$train_portion}"
master_port="${3:-$master_port}"

echo seed: $seed, train_portion: $train_portion, master_port: $master_port

random_time=$(shuf -i 10-60 -n 1)

echo "Waiting for $random_time seconds..."
sleep "$random_time"

python -m torch.distributed.launch --nproc_per_node=4 --master_port ${master_port} --use_env search_spaces/nanoGPT/train_search.py config=search_spaces/nanoGPT/config/train_tinystories_drnas.py --train_portion=${train_portion} --seed=${seed}
conda deactivate
