#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
n_epoch=$6

python3 scripts/experiments/influence.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --trunc_frac $trunc_frac \
  --update_set $update_set \
  --n_epoch $n_epoch \
