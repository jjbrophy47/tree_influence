dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
inf_obj=$6
global_op=$7
mem=$8
time=$9
partition=${10}

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

    job_name=CI_${dataset}_${tree_type}_${method}_${inf_obj}_${global_op}

    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/ci/$job_name \
           --error=jobs/errors/ci/$job_name \
           jobs/ci/runner.sh $dataset $tree_type $method \
           $inf_obj $trunc_frac $update_set $global_op

done
