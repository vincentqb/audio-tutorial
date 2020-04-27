#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%j.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%j.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=learnfair
#SBATCH --time=4320
#SBATCH --mem-per-cpu=5120
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
# 2x (number of data workers + number of GPUs requested)

arch=$1
bs=$2
lr=$3

# PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_JOB_NODELIST:0:9}${SLURM_JOB_NODELIST:10:4}
export MASTER_PORT=29500

srun --label \
    python /private/home/vincentqb/experiment/PipelineTrain.py \
	--arch $arch --batch-size $bs --learning-rate $lr \
	--resume /private/home/vincentqb/experiment/checkpoint-$SLURM_JOB_ID-$arch-$bs-$lr.pth.tar
	# --distributed --world-size $SLURM_JOB_NUM_NODES --dist-url 'env://' --dist-backend='nccl'
