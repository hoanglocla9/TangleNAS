#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J small_autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env search_spaces/AutoFormer/supernet_train_inherit.py --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --patch_size 16  --epochs 1500 --warmup-epochs 20 --output gdas_imnet_fast_finetune/ --batch-size 64 --amp --model_path /work/dlclarge2/sukthank-tanglenas/TangleNAS-dev/output_imagenet_gdas_we2/checkpoint_78.pth  --df_path /work/dlclarge2/sukthank-tanglenas/TangleNAS-dev/output_imagenet_gdas_we2/arch_trajectory.pkl --lr 1e-5 --warmup-lr 1e-4 --min-lr 1e-6
