#!/bin/bash
#SBATCH --job-name=Multi_Test
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

if [[ $exp = 'remove_set' ]]; then
    python3 scripts/experiments/multi_test/remove_set.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method

elif [[ $exp = 'label_set' ]]; then
    python3 scripts/experiments/multi_test/label_set.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method

elif [[ $exp = 'poison_set' ]]; then
    python3 scripts/experiments/multi_test/poison_set.py \
      --dataset $dataset \
      --tree_type $tree_type \
      --method $method
fi
