skip=$1
dataset=$2
tree_type=$3
n_estimators=$4
max_depth=$5
method=$6
trunc_frac=$7
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

    job_name=RR_${dataset}_${tree_type}_${method}_${inf_obj}_${global_op}

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/roar/$job_name \
           --error=jobs/errors/roar/$job_name \
           jobs/roar/runner.sh $skip $dataset $tree_type \
           $n_estimators $max_depth $method \
           $inf_obj $trunc_frac $global_op

done
