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
python spos_re.py --model_path /path/to/out_search_spos_0.8_9001_6000_20230828-174230/latest_ckpt.pt --train_portion 0.8
