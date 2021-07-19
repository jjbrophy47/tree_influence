dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
trunc_frac=$6
update_set=$7
global_op=$8
inf_op=$9
mem=${10}
time=${11}
partition=${12}

rs_list=(1 2 3 4 5)

if [ $inf_op = 0 ]
then
    inf_obj_list=('global')
elif [ $inf_op = 1 ]
then
    inf_obj_list=('local')
else
    inf_obj_list=('global' 'local')
fi

for inf_obj in ${inf_obj_list[@]}; do

    job_name=CI_${dataset}_${tree_type}_${method}_${inf_obj}_${global_op}

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/ci/$job_name \
           --error=jobs/errors/ci/$job_name \
           jobs/ci/runner.sh $dataset $tree_type \
           $n_estimators $max_depth $method \
           $inf_obj $trunc_frac $update_set $global_op

done
