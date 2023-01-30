#!/bin/bash
#SBATCH -p partition_name
#SBATCH --gres=gpu:8
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 search_spaces/SwinIR/main_train_psnr.py --opt search_spaces/SwinIR/options/train_swinir_sr_lightweight_darts.json  --dist True
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 search_spaces/SwinIR/main_train_psnr.py --opt search_spaces/SwinIR/options/train_swinir_sr_lightweight_drnas.json  --dist True
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 search_spaces/SwinIR/main_train_psnr.py --opt search_spaces/SwinIR/options/train_swinir_sr_lightweight_gdas.json  --dist True