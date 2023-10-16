#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J charlm # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas --train_portion 0.5
python toy_search_spaces/conv_macro/train_spos.py --train_portion 0.5 --seed 9004
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas --train_portion 0.5
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas --train_portion 0.5
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas --train_portion 0.5
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas --train_portion 0.5
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type drnas
#python toy_search_spaces/conv_macro/train_search.py --optimizer_type darts_v1
