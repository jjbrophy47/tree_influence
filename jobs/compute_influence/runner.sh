#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
inf_obj=$6

python3 scripts/experiments/compute_influence.py \
  --dataset $dataset \
  --tree_type $type \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --inf_obj $inf_obj \
