dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
n_epoch=$6
ncpu=$7
time=$8
partition=$9

job_name=I_${dataset}_${tree_type}_${method}

sbatch --cpus-per-task=$ncpu \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/inf/$job_name \
       --error=jobs/errors/inf/$job_name \
       jobs/inf/runner.sh $dataset $tree_type $method $trunc_frac $update_set $n_epoch
