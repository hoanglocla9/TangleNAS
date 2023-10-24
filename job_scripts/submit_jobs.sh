#!/bin/bash

# Check if the argument is provided
if [ $# -ne 1 ]; then
    echo "Optimizer not specified. Usage: submit_jobs.sh <drnas/spos>"
    exit 1
fi

# Get the first command line argument
opt="$1"

# Verify if the fruit is "apple" or "orange"
if [ "$opt" == "drnas" ]; then
    echo "Selected optimizer: drnas"
elif [ "$opt" == "spos" ]; then
    echo "Selected optimzer: spos"
else
    echo "Error: Invalid optimizer. Please choose either 'drnas' or 'spos'."
    exit 1
fi

seeds=(9001)
trainportions=(0.5 0.8)
opt=$1
master_port=19001

for seed in "${seeds[@]}"
do
    for portion in "${trainportions[@]}"
    do
        sbatch ./job_scripts/train_nanogpt_${opt}.sh $seed $portion $master_port
        master_port=$((master_port + 1))
    done
done
