dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
trunc_frac=$6
ncpu=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)
inf_obj_list=('global' 'local')

for inf_obj in ${inf_obj_list[@]}; do

    job_name=CI_${dataset}_${tree_type}_${method}_${inf_obj}

    sbatch --cpus-per-task=$ncpu \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/compute_influence/$job_name \
           --error=jobs/errors/compute_influence/$job_name \
           jobs/compute_influence/runner.sh $dataset $tree_type \
           $n_estimators $max_depth $method $inf_obj $trunc_frac

done
