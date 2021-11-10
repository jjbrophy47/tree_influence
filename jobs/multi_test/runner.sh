#!/bin/bash
#SBATCH --job-name=Single_Test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

exp=$1
tree_type=$2
method=$3

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

if [[ $exp = 'remove' ]]; then
    python3 scripts/experiments/single_test/remove.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method

elif [[ $exp = 'label' ]]; then
    python3 scripts/experiments/single_test/label.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method

elif [[ $exp = 'poison' ]]; then
    python3 scripts/experiments/single_test/poison.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method
fi
