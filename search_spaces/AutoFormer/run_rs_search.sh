#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J RS_OneShotWE_prior # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recieve mails about end and timeouts/crashes of your job)
#SBATCH -a 0-5

input_sizes=(32 32 32 224 224 224)
patch_sizes=(2 2 2 16 16 16)
optimizers=(darts drnas gdas darts drnas gdas)
dataset=CIFAR100
dataset_filename=cifar100

# for SLURM_ARRAY_TASK_ID in {3..5}; do

input_size=${input_sizes[$SLURM_ARRAY_TASK_ID]}
patch_size=${patch_sizes[$SLURM_ARRAY_TASK_ID]}
optimizer=${optimizers[$SLURM_ARRAY_TASK_ID]}

filepath=./checkpoints/checkpoint_${optimizer}_${dataset_filename}_${input_size}.pth

echo python random_search.py --batch-size 32 --dataset-subset-size 33 --cfg no_config_file --device cpu --search_iters 2000 --input-size $input_size --patch_size $patch_size --one-shot-optimizer $optimizer --model-path $filepath
python random_search.py --batch-size 32 --dataset-subset-size 10000 --cfg no_config_file --device cuda --search_iters 2000 --input-size $input_size --patch_size $patch_size --one-shot-optimizer $optimizer --model-path $filepath --sample-with-priors

# done
