#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
inf_obj=$6
trunc_frac=$7

python3 scripts/experiments/roar.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --inf_obj $inf_obj \
  --trunc_frac $trunc_frac \
