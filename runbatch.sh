#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%A-%a.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%A-%a.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=learnfair
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=5120
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --array=1-16
# number of CPUs = 2x (number of data workers + number of GPUs requested)

if [[ "$SLURM_ARRAY_TASK_COUNT" -ne $((2 * 3 * 3)) ]]; then
    exit
fi

i=1
for arch in 'wav2letter' 'lstm'; do
    for bs in 128 256 512; do
        for lr in 1. .01 .0001; do
            if [[ "$i" == "$SLURM_ARRAY_TASK_ID" ]]; then break; fi
            ((i++))
        done
        if [[ "$i" == "$SLURM_ARRAY_TASK_ID" ]]; then break; fi
    done
    if [[ "$i" == "$SLURM_ARRAY_TASK_ID" ]]; then break; fi
done

echo $SLURM_JOB_ID $arch $bs $lr

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_JOB_NODELIST:0:9}${SLURM_JOB_NODELIST:10:4}
export MASTER_PORT=29500

# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

srun --label \
    python /private/home/vincentqb/experiment/PipelineTrain.py \
	--arch $arch --batch-size $bs --learning-rate $lr \
	--resume /private/home/vincentqb/experiment/checkpoint-$SLURM_JOB_ID-$arch-$bs-$lr.pth.tar \
	--world-size $SLURM_JOB_NUM_NODES --dist-url 'env://' --dist-backend='nccl'
