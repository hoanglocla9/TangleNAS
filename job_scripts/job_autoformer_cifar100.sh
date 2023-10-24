#!/bin/bash
#SBATCH -p single
#SBATCH --gres=gpu:2
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=150GB # memory to allocate
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
export PYTHONPATH=.
python -m torch.distributed.launch --nproc_per_node=2 --use_env search_spaces/AutoFormer/supernet_train_cifar10.py --data-path . --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 --output "output_autoformer_drnas_cifar100" --batch-size 512 --one_shot_opt drnas --use_we_v2 --data-set CIFAR100 --ratio 0.8
