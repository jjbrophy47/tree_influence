#!/bin/bash
#SBATCH --job-name=Targeted_Edit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

tree_type=$1
method=$2

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/single_test/targeted_edit.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method
