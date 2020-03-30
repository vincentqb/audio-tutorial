#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%j.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%j.err
#SBATCH --partition=scavenge
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --open-mode=append
#SBATCH --time=40:00:00
#SBATCH --signal=USR1@600


# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=29500

# delelte the second line to start single node training
srun --label python /private/home/vincentqb/experiment/PipelineTrain.py --batch-size 1024 \
	--resume /private/home/vincentqb/experiment/checkpoint-batch.pth.tar \
	--world-size $SLURM_NNODES --dist-url 'env://' --dist-backend='nccl'
