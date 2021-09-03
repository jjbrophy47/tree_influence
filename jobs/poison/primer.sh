dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
ncpu=$6
time=$7
partition=$8

job_name=P_${dataset}_${tree_type}_${method}

sbatch --cpus-per-task=$ncpu \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/poison/$job_name \
       --error=jobs/errors/poison/$job_name \
       jobs/poison/runner.sh $dataset $tree_type $method $trunc_frac $update_set
