#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 4 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
python -m torch.distributed.launch --nproc_per_node=8 --use_env search_spaces/AutoFormer/supernet_train.py --data-path . --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 --output "output_autoformer_darts_0.5_smaller_lr" --batch-size 64 --one_shot_opt darts_v1 --use_we_v2 --amp --lr 1e-4 --warmup-lr 1e-6 --min-lr 1e-5
