#!/bin/bash
#SBATCH -p single
#SBATCH --gres=gpu:4
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB # memory to allocate
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
export PYTHONPATH=.
python -m torch.distributed.launch --nproc_per_node=4 --use_env search_spaces/MobileNetV3/finetune/mobilenet_finetune.py --one_shot_opt drnas --opt_strategy "alternating"
#torchrun --standalone --nproc_per_node=2 toy_search_spaces/nanoGPT/train_search.py --config toy_search_spaces/nanoGPT/config/train_gpt2.py
#torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py
