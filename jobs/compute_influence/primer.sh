dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5)
inf_obj_list=('global' 'local')

for inf_obj in ${inf_obj_list[@]}; do

    job_name=CI_${dataset}_${method}_${inf_obj}

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/compute_influence/$job_name \
           --error=jobs/errors/compute_influence/$job_name \
           jobs/compute_influence/runner.sh $dataset $tree_type \
           $n_estimators $max_depth $method $inf_obj

done
