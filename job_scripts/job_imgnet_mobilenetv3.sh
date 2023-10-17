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
#python -m torch.distributed.launch --nproc_per_node=8 --use_env search_spaces/AutoFormer/supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 --output "output_autoformer_drnas_imgnet_4_gpus" --batch-size 64 --one_shot_opt drnas --use_we_v2 --amp --data-set IMNET --resume /work/dlclarge2/sukthank-tanglenas/Cream/checkpoint_421.pth
#python -m torch.distributed.launch --nproc_per_node=8 --use_env search_spaces/MobileNetV3/search/mobilenet_search_base.py --one_shot_opt drnas --opt_strategy "alternating"  --valid_size 0.2
#python -m search.experiment_search --dataset cifar10 --optimizer drnas --searchspace nb201  --seed 9001 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.8
#python -m search.experiment_search --dataset cifar10 --optimizer drnas --searchspace nb201  --seed 9002 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.8
#python -m search.experiment_search --dataset cifar10 --optimizer drnas --searchspace nb201  --seed 9003 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar10 --optimizer drnas --searchspace nb201  --seed 9004 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar100 --optimizer drnas --searchspace nb201  --seed 9004 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar100 --optimizer drnas --searchspace nb201  --seed 9004 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar100 --optimizer drnas --searchspace nb201  --seed 9004 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar100 --optimizer drnas --searchspace nb201  --seed 9004 --submission --use_we_v2  --path_to_benchmark /work/dlclarge2/sukthank-tanglenas/NAS-Bench-201-v1_0-e61699.pth --train_portion 0.5 
#python -m search.experiment_search --dataset cifar10 --optimizer drnas --searchspace darts --seed 9004   --use_we_v2
