#!/bin/bash
#SBATCH --job-name=Reinf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

tree_type=$1
method=$2
strategy=$3
n_early_stop=$4

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/single_test/reinfluence.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --strategy $strategy \
  --n_early_stop $n_early_stop \
