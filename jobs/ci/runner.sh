#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$1
tree_type=$2
method=$3
inf_obj=$4
trunc_frac=$5
update_set=$6
local_op=$7
global_op=$8

python3 scripts/experiments/compute_influence.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --inf_obj $inf_obj \
  --trunc_frac $trunc_frac \
  --update_set $update_set \
  --local_op $local_op \
  --global_op $global_op \
