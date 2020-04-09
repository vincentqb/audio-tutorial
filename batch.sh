#! /bin/bash

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=29500

for arch in 'wav2letter' 'lstm'; do
    for bs in 1 256 2048; do
        for lr in 1. .01 .0001; do
            sbatch /private/home/vincentqb/experiment/run.sh $arch $bs $lr
        done;
    done;
done;
