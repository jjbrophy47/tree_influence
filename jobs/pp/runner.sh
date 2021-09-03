#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$1
model=$2
tune_frac=$3
train_frac=$4

python3 scripts/experiments/prediction.py \
  --dataset $dataset \
  --model $model \
  --tune_frac $tune_frac \
  --train_frac $train_frac \
