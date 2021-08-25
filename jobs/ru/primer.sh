dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
ncpu=$6
time=$7
partition=$8

job_name=RU_${dataset}_${tree_type}_${method}

sbatch --cpus-per-task=$ncpu \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/ru/$job_name \
       --error=jobs/errors/ru/$job_name \
       jobs/ru/runner.sh $dataset $tree_type $method \
       $trunc_frac $update_set
