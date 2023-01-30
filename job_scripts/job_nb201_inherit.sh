#!/bin/bash
#SBATCH -p partition_name
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
python -m train.experiment_train --searchspace nb201 --finetune --seed 9001 --model_path path/to/supernet --path_to_benchmark NAS-Bench-201-v1_0-e61699.pth