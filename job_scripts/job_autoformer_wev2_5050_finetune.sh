#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --exclude=dlcgpu36
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer-wev2-finetune # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH -a 9001-9004 # array values

source ~/.bash_profile
conda activate tanglenas

export PYTHONPATH=$PYTHONPATH:/work/dlclarge1/krishnan-TangleNAS/TangleNAS-dev

optimizer=$1
dataset=$2

picklefile="/work/dlclarge1/krishnan-TangleNAS/TangleNAS-dev/pretrained_models/50_50_ckpts/${optimizer}_${dataset}/arch_trajectory.pkl"
pthfile="/work/dlclarge1/krishnan-TangleNAS/TangleNAS-dev/pretrained_models/50_50_ckpts/${optimizer}_${dataset}/checkpoint_476.pth"

ls $picklefile
echo "pickle file found!"

ls $pthfile
echo pth file found!

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1723 --use_env search_spaces/AutoFormer/supernet_train_inherit.py --data-set $dataset --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --patch_size 16  --epochs 300 --warmup-epochs 20 --output autoformer_wev2_${optimizer}_${dataset}_${SLURM_ARRAY_TASK_ID}_5050_finetune/ --batch-size 64 --amp --model_path $pthfile --df_path $picklefile --arch_epoch 476 --lr 1e-5 --warmup-lr 1e-4 --min-lr 1e-6 --seed $SLURM_ARRAY_TASK_ID

echo "Finish"