dataset=$1
tree_type=$2
n_estimators=$3
max_depth=$4
method=$5
trunc_frac=$6
global_op=$7
inf_op=$8
mem=$9
time=${10}
partition=${11}

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

    job_name=RR_${dataset}_${tree_type}_${method}_${inf_obj}_${global_op}

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/roar/$job_name \
           --error=jobs/errors/roar/$job_name \
           jobs/roar/runner.sh $dataset $tree_type \
           $n_estimators $max_depth $method \
           $inf_obj $trunc_frac $global_op

done
