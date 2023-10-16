#!/bin/bash
#SBATCH -p single
#SBATCH --gres=gpu:2
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 24 # number of cores
#SBATCH --cpus-per-task=24
#SBATCH --mem=150GB # memory to allocate
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
export PYTHONPATH=.
torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py
#torchrun --standalone --nproc_per_node=2  train.py config/train_gpt2.py
