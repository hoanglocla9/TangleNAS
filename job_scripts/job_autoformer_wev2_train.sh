#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer-wev2-train # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)

source ~/.bash_profile
conda activate tanglenas

export PYTHONPATH=$PYTHONPATH:/work/dlclarge1/krishnan-TangleNAS/TangleNAS-dev

picklefile="pretrained_models/autoformer/arch_trajectory_drnas_cifar100_we2.pkl"
python -m torch.distributed.launch --nproc_per_node=8  --master_port=1723 --use_env search_spaces/AutoFormer/supernet_from_scratch.py --data-set CIFAR100 --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --patch_size 16  --epochs 500 --warmup-epochs 20 --output autoformer_wev2_drnas_cf100_train/ --batch-size 64 --amp  --df_path $picklefile  --arch_epoch 750  --seed 9001
echo "Finish"