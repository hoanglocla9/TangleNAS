#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 4 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python spos_rs.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.5_9001_6000_20230829-103744/latest_ckpt.pt 
#python spos_rs.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.5_9006_6000_20230901-135003/latest_ckpt.pt 
python spos_rs.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.5_9011_6000_20230831-161307/latest_ckpt.pt 
#python spos_rs.py --model_path /work/dlclarge2/sukthank-llama/tanglenas_checkpoints/all_models/out_search_spos_0.5_9016_6000_20230901-143435/latest_ckpt.pt 

