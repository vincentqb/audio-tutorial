#! /bin/bash

for arch in 'wav2letter' 'lstm'; do
    for bs in 256 2048; do
        for lr in 1. .01 .0001; do
            sbatch /private/home/vincentqb/experiment/run.sh $arch $bs $lr
        done;
    done;
done;
