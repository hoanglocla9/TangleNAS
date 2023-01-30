#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J small_autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
python train_search.py --unrolled
#python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --patch_size 2  --epochs 1000 --warmup-epochs 20 --output output_gdas_no_prior_cifar100/ --batch-size 32  --amp #--prior
#python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --patch_size 2  --epochs 1000 --warmup-epochs 20 --output output_cifar100_correct/ --batch-size 8 --amp #--data-path /work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/
#python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --patch_size 16  --epochs 1000 --warmup-epochs 20 --output output_imnet_drnas/ --batch-size 16 --amp  --one_shot_opt drnas # --prior
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 SwinIRWE/main_train_psnr.py --opt SwinIRWE/options/swinir/train_swinir_sr_lightweight.json  --dist True
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=1783 --use_env search_spaces/AutoFormer/supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --epochs 200 --warmup-epochs 20 --output output_cifar10_darts_only_mlp_full/ --batch-size 16 --amp  --one_shot_opt darts
#python -m torch.distributed.launch --nproc_per_node=4  --master_port=1723 --use_env supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --patch_size 2  --epochs 1000 --warmup-epochs 20 --output output_cifar100_gdas_prior/ --batch-size 32 --amp  --one_shot_opt gdas --prior #--data-path /work/dlclarge2/sukthank-naslib_one_shot/Cream/AutoFormer/imagenet/
