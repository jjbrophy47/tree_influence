dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
ncpu=$6
time=$7
partition=$8

seed_list=(1 2 3 4 5)

for seed in ${seed_list[@]}; do

    job_name=RU_${dataset}_${tree_type}_${method}_${seed}

    sbatch --cpus-per-task=$ncpu \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/ru/$job_name \
           --error=jobs/errors/ru/$job_name \
           jobs/ru/runner2.sh $dataset $tree_type $method \
           $trunc_frac $update_set $seed

done
