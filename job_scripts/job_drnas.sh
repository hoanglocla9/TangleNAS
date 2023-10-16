#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J small_autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python search_spaces/CharLM/search/search.py --mixop darts_v1 --portion 0.5 --use_we_v2
#python search_spaces/CharLM/search/search.py --mixop drnas --portion 0.5 --use_we_v2
python toy_search_spaces/nanoGPT/train_search.py --config toy_search_spaces/nanoGPT/config/train_shakespeare_char.py --optimizer drnas  --seed 9001 --search_depth --search_emb --search_ratio --search_heads
python toy_search_spaces/nanoGPT/train_search.py --config toy_search_spaces/nanoGPT/config/train_shakespeare_char.py --optimizer drnas  --seed 9002 --search_depth --search_emb --search_ratio --search_heads
python toy_search_spaces/nanoGPT/train_search.py --config toy_search_spaces/nanoGPT/config/train_shakespeare_char.py --optimizer drnas  --seed 9003 --search_depth --search_emb --search_ratio --search_heads
python toy_search_spaces/nanoGPT/train_search.py --config toy_search_spaces/nanoGPT/config/train_shakespeare_char.py --optimizer drnas  --seed 9004 --search_depth --search_emb --search_ratio --search_heads
