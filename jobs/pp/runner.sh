#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$1
model=$2

python3 scripts/experiments/predictive_performance.py \
  --dataset $dataset \
  --model $model \
