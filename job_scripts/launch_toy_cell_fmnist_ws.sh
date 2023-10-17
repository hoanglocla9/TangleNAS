#!/bin/bash
optimizers=("drnas")
datasets=("fashion_mnist")
seed=(9001 9002 9003 9004)
entanglements=("ws")
submission=0
use_we_v2=0
for s in "${seed[@]}"
do
 for ent in "${entanglements[@]}"
 do
    for dataset in "${datasets[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do

            we_version=""
            # use_we_v2 overrides the --no_weight_entanglement flag, so turn it on only for we
            if [ "$ent" == "we" ]; then
                we_v2_flag=$use_we_v2

                if [ $use_we_v2 -eq 1 ]; then
                    we_version="-v2"
                else
                    we_version="-v1"
                fi
            else
                we_v2_flag=0
            fi

            exp_name="toy_cell_space-${optimizer}-${dataset}-${ent}${we_version}"
            echo Submitting job $exp_name
            echo "sbatch -J $exp_name job_scripts/job_experiment_ws.sh $dataset $optimizer toy_cell $ent $we_v2_flag $submission $s"
            sbatch job_scripts/job_experiment_ws.sh $dataset $optimizer "toy_cell" $ent $we_v2_flag $submission $s
        done
    done
 done
done