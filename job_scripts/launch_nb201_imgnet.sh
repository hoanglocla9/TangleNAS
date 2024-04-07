#!/bin/bash
optimizers=("spos" "drnas")
datasets=("imgnet16_120")
entanglements=("we")
train_portion=(0.8 0.5)
submission=1
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

            exp_name="nb201-${optimizer}-${dataset}-${ent}${we_version}"
            echo Submitting job $exp_name

            sbatch -J $exp_name job_scripts/job_experiment_nb201_imgnet.sh $dataset $optimizer "nb201" $ent $we_v2_flag $submission $t
        done
    done
 done
done
