#!/bin/bash
#SBATCH --job-name=Inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

method=$1
tree_type=$2

declare -A datasets
datasets[1]='adult'
datasets[2]='bank_marketing'
datasets[3]='bean'
datasets[4]='compas'
datasets[5]='concrete'
datasets[6]='credit_card'
datasets[7]='diabetes'
datasets[8]='energy'
datasets[9]='flight_delays'
datasets[10]='german_credit'
datasets[11]='htru2'
datasets[12]='life'
datasets[13]='naval'
datasets[14]='no_show'
datasets[15]='obesity'
datasets[16]='power'
datasets[17]='protein'
datasets[18]='spambase'
datasets[19]='surgical'
datasets[20]='twitter'
datasets[21]='vaccine'
datasets[22]='wine'

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/influence.py \
  --method $method \
  --dataset $dataset \
  --tree_type $tree_type \
