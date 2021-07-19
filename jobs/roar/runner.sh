#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

skip=$1
dataset=$2
tree_type=$3
n_estimators=$4
max_depth=$5
method=$6
inf_obj=$7
trunc_frac=$8
update_set=$9
global_op=${10}

python3 scripts/experiments/roar.py \
  --skip $skip \
  --dataset $dataset \
  --tree_type $tree_type \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --inf_obj $inf_obj \
  --trunc_frac $trunc_frac \
  --update_set $update_set \
  --global_op $global_op \
