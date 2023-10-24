#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J charlm # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python search_spaces/CharLM/search.py --mixop gdas --batch-size 64 --use_we_v2
python spos_re.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.8_9001_6000_20230828-174230/latest_ckpt.pt --train_portion 0.8
#python spos_re.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.8_9006_6000_20230831-151435/latest_ckpt.pt --train_portion 0.8
#python spos_re.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.8_9011_6000_20230831-174737/latest_ckpt.pt --train_portion 0.8
#python spos_re.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.8_9016_6000_20230831-181755/latest_ckpt.pt --train_portion 0.8
#python toy_search_spaces/nanoGPT/spos_rs.py --model_path /work/dlclarge2/sukthank-tanglenas/merge_with_main/TangleNAS-dev/output_charlm/out_train_spos_spos_9004_0.8_10000_20230805-170318/ckpt.pt --train_portion 0.8
##python toy_search_spaces/nanoGPT/spos_rs.py --model_path /work/dlclarge2/sukthank-tanglenas/merge_with_main/TangleNAS-dev/output_charlm/out_train_spos_spos_9002_0.5_10000_20230805-170618/ckpt.pt --train_portion 0.5
#python toy_search_spaces/nanoGPT/spos_rs.py --model_path /work/dlclarge2/sukthank-tanglenas/merge_with_main/TangleNAS-dev/output_charlm/out_train_spos_spos_9001_0.8_10000_20230805-170310/ckpt.pt --train_portion 0.8
#python toy_search_spaces/nanoGPT/spos_rs.py --model_path /work/dlclarge2/sukthank-tanglenas/merge_with_main/TangleNAS-dev/output_charlm/out_train_spos_spos_9001_0.5_10000_20230805-170310/ckpt.pt --train_portion 0.5

#python search_spaces/CharLM/search.py --batch-size 16 --mixop drnas
#python search_spaces/CharLM/train_spos.py --mixop spos --batch-size 64
#python search_spaces/CharLM/train.py --n_embed 256 --n_layers 6 --num_heads 4 8 8 8 8 8 --mlp_ratio 4 4 4 4 4 4 # darts simul
#python search_spaces/CharLM/train.py --n_embed 256 --n_layers 6 --num_heads 4 8 8 8 8 8 --mlp_ratio 4 4 4 4 4 4 # drnas simul
#python search_spaces/CharLM/train.py --n_embed 96  --n_layers 2 --num_heads 2 2 2 2 8 4 --mlp_ratio 1 1 2 2 4 2 # gdas simul
#python search_spaces/CharLM/train.py --n_embed 96 --n_layers 2 --num_heads 8 8 2 2 4 2 --mlp_ratio 2 1 1 1 1 1 # darts alt
#python search_spaces/CharLM/train.py --n_embed 96 --n_layers 2 --num_heads 4 2 2 2 2 8 --mlp_ratio 4 2 1 1 1 2 # drnas alt
#python search_spaces/CharLM/train.py --n_embed 96  --n_layers 4 --num_heads 2 2 2 2 2 2 --mlp_ratio 1 1 1 1 1 1 # gdas alt
#python -m torch.distributed.launch --nproc_per_node=6  --master_port=1723 --use_env search_spaces/AutoFormer/supernet_train.py  --gp --change_qkv --relative_position --mode super --dist-eval --cfg search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml  --patch_size 16  --epochs 500 --warmup-epochs 20 --output output_imagenet_darts/ --batch-size 128 --amp  --one_shot_opt  darts_v1
