#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J nb201 # sets the job name. If not specified, the file name will be used as job name
dataset=$1
optimizer=$2
search_space=$3
seed=$4
python -m search.experiment_search  --dataset ${dataset}  --optimizer ${optimizer}  --searchspace ${search_space} --seed ${seed} --path_to_benchmark /work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth 
#/work/dlclarge1/sukthank-transformer_search/GraViT-E/main/OneShotNASwithWE/data/NATS-sss-v1_0-50262-simple #/work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth
#python -m search.experiment_search --dataset cifar10  --optimizer spos  --searchspace nb201 --seed 9004 --path_to_benchmark /work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth #--path_to_benchmark /work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth #/work/dlclarge1/sukthank-transformer_search/GraViT-E/main/OneShotNASwithWE/data/NATS-sss-v1_0-50262-simple #/work/dlclarge1/sukthank-transformer_search/GraViT-E/experiments/OneShotNASwithWE/NATS-sss-v1_0-50262-simple #/work/dlclarge1/sukthank-transformer_search/reproduce_oneshot/DrNAS/201-space/NAS-Bench-201-v1_0-e61699.pth #--data_path data/ImageNet16/

