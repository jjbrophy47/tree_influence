skip=$1
dataset=$2
tree_type=$3
method=$4
trunc_frac=$5
update_set=$6
inf_obj=$7
local_op=$8
global_op=$9
ncpu=${10}
time=${11}
partition=${12}

rs_list=(1 2 3 4 5)

if [ $inf_obj = 'global' ]
then
    inf_obj_list=('global')
elif [ $inf_obj =  'local' ]
then
    inf_obj_list=('local')
else
    inf_obj_list=('global' 'local')
fi

for inf_obj in ${inf_obj_list[@]}; do

    job_name=RR_${dataset}_${tree_type}_${method}_${inf_obj}_${local_op}_${global_op}

    sbatch --cpus-per-task=$ncpu \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/roar/$job_name \
           --error=jobs/errors/roar/$job_name \
           jobs/roar/runner.sh $skip $dataset $tree_type $method \
           $inf_obj $trunc_frac $update_set $local_op $global_op

done
