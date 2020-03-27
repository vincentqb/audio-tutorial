#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%j.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%j.err
#SBATCH --partition=learnfair
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --open-mode=append
#SBATCH --time=40:00:00

srun --label python /private/home/vincentqb/experiment/PipelineTrain.py --batch-size 1024 --checkpoint "checkpoint-batch.pth.tar"
