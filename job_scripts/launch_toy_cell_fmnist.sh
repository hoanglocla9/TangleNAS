#!/bin/bash
optimizers=("drnas" "darts_v1" "gdas" "darts_v2")
datasets=("fashion_mnist")
entanglements=("we")
submission=0
use_we_v2=1

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

            exp_name="nb201-${optimizer}-${dataset}-${ent}${we_version}"
            echo Submitting job $exp_name

            sbatch -J $exp_name job_scripts/job_experiment.sh $dataset $optimizer "toy_cell_stacked" $ent $we_v2_flag $submission
        done
    done
done