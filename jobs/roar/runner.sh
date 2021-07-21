#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

skip=$1
dataset=$2
tree_type=$3
method=$4
inf_obj=$5
trunc_frac=$6
update_set=$7
global_op=$8

python3 scripts/experiments/roar.py \
  --skip $skip \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --inf_obj $inf_obj \
  --trunc_frac $trunc_frac \
  --update_set $update_set \
  --global_op $global_op \
