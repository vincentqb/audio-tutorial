#! /bin/bash

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=29500

for arch in 'wav2letter' 'lstm'; do
	for bs in 1 10 100; do
		for lr in 1 .1 .01 .001; do
			for wd in 1e-2 1e-4 1e-6; do
				sbatch /private/home/vincentqb/experiment/run.sh $arch $bs $lr $wd
			done;
		done;
	done;
done;
