#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --qos=c2
#SBATCH --mem=128gb
#SBATCH -t 7-0:00:00
#SBATCH -o jobs_logs/job.%J.out
#SBATCH -e jobs_logs/job.%J.err


CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/nb3891/.conda/envs/contrastive_rl
conda activate $CONDA_ENV

python train.py --dataset cifar10 --simclr_bs 512 --mode async