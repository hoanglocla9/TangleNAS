#!/bin/bash
#SBATCH -p partiiton_name
#SBATCH --gres=gpu:8
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env search_spaces/AutoFormer/supernet_train_inherit.py  --gp  --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml --patch_size 16  --epochs 1000 --warmup-epochs 20 --output output_inherit_imnet_drnas/ --batch-size 64 --amp --change_qkv