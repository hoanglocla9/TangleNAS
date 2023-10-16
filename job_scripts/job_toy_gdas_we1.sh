#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J toy # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python search_spaces/CharLM/search.py --mixop gdas --batch-size 64 --use_we_v2
python toy_search_space/gdas_we1_search.py --seed 9001 --portion 0.8
python toy_search_space/gdas_we1_search.py --seed 9002 --portion 0.8
python toy_search_space/gdas_we1_search.py --seed 9003 --portion 0.8
python toy_search_space/gdas_we1_search.py --seed 9004 --portion 0.8
python toy_search_space/gdas_we1_search.py --seed 9001 --portion 0.5
python toy_search_space/gdas_we1_search.py --seed 9002 --portion 0.5
python toy_search_space/gdas_we1_search.py --seed 9003 --portion 0.5
python toy_search_space/gdas_we1_search.py --seed 9004 --portion 0.5
#python search_spaces/CharLM/search.py --batch-size 64 --mixop gdas --portion 0.8 --seed 9001
#python search_spaces/CharLM/search.py --batch-size 16 --mixop drnas
#python search_spaces/CharLM/train_spos.py --mixop spos --batch-size 64
#python search_spaces/CharLM/train.py --n_embed 256 --n_layers 6 --num_heads 4 8 8 8 8 8 --mlp_ratio 4 4 4 4 4 4 # darts simul
#python search_spaces/CharLM/train.py --n_embed 256 --n_layers 6 --num_heads 4 8 8 8 8 8 --mlp_ratio 4 4 4 4 4 4 # drnas simul
#python search_spaces/CharLM/train.py --n_embed 96  --n_layers 2 --num_heads 2 2 2 2 8 4 --mlp_ratio 1 1 2 2 4 2 # gdas simul
#python search_spaces/CharLM/train.py --n_embed 96 --n_layers 2 --num_heads 8 8 2 2 4 2 --mlp_ratio 2 1 1 1 1 1 # darts alt
#python search_spaces/CharLM/train.py --n_embed 96 --n_layers 2 --num_heads 4 2 2 2 2 8 --mlp_ratio 4 2 1 1 1 2 # drnas alt
#python search_spaces/CharLM/train.py --n_embed 96  --n_layers 4 --num_heads 2 2 2 2 2 2 --mlp_ratio 1 1 1 1 1 1 # gdas alt
#python -m torch.distributed.launch --nproc_per_node=6  --master_port=1723 --use_env search_spaces/AutoFormer/supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --patch_size 16  --epochs 500 --warmup-epochs 20 --output output_imagenet_darts/ --batch-size 128 --amp  --one_shot_opt  darts_v1
