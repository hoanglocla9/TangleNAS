#!/bin/bash
optimizers=("drnas")
datasets=("cifar10")
entanglements=("we")
train_portion=(0.5 0.8)
submission=0
use_we_v2=1
for t in "${train_portion[@]}"
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

            exp_name="darts-${optimizer}-${dataset}-${ent}${we_version}${t}"
            echo Submitting job $exp_name

            sbatch -J $exp_name job_scripts/job_experiment_darts.sh $dataset $optimizer "darts" $ent $we_v2_flag $submission $t
        done
    done
 done
done
