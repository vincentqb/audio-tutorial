#! /bin/bash

arch=$1
bs=$2
lr=$3
wd=$4

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%j.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%j.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --ntasks-per-node=1
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=5120

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=29500

srun --label python /private/home/vincentqb/experiment/PipelineTrain.py \
	--arch $arch --batch-size $bs --learning-rate $lr --weight-decay $wd \
	--resume /private/home/vincentqb/experiment/checkpoint-$arch-$bs-$lr-$wd.pth.tar
	# --world-size $SLURM_NNODES --dist-url 'env://' --dist-backend='nccl'
