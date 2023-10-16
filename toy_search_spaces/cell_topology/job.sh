#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J small_autoformer # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#python search_spaces/CharLM/search/search.py --mixop darts_v1 --portion 0.5 --use_we_v2
#python search_spaces/CharLM/search/search.py --mixop drnas --portion 0.5 --use_we_v2
python create_benchmark.py --start_index 0 --end_index 100 --benchmark_file_name benchmark_dictionary_1.pkl
#python create_benchmark_2.py #--use_we_v2
#python create_benchmark_3.py #--use_we_v2
#python create_benchmark_4.py #--use_we_v2
#python create_benchmark_5.py #--use_we_v2
#python create_benchmark_6.py #--use_we_v2
#python create_benchmark_7.py #--use_we_v2
#python create_benchmark_8.py #--use_we_v2
#python create_benchmark_9.py #--use_we_v2
#python create_benchmark_10.py #--use_we_v2
#python create_benchmark_11.py #--use_we_v2
#python create_benchmark_12.py #--use_we_v2
#python create_benchmark_13.py #--use_we_v2
#python create_benchmark_14.py #--use_we_v2
#python create_benchmark_15.py #--use_we_v2
#python create_benchmark_16.py #--use_we_v2
#python create_benchmark_17.py #--use_we_v2
#python create_benchmark_18.py #--use_we_v2
#python create_benchmark_19.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_21.py #--use_we_v2
#python create_benchmark_22.py #--use_we_v2
#python create_benchmark_23.py #--use_we_v2
#python create_benchmark_24.py #--use_we_v2
#python create_benchmark_40.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
#python create_benchmark_20.py #--use_we_v2
