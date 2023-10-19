#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 12 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
python -m torch.distributed.launch --nproc_per_node=8 --use_env search_spaces/MobileNetV3/finetune/mobilenet_finetune.py --one_shot_opt drnas --opt_strategy "alternating"  --valid_size 10000