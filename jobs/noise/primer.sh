dataset=$1
tree_type=$2
method=$3
trunc_frac=$4
update_set=$5
strategy=$6
ncpu=$7
time=$8
partition=$9

noise_frac_list=(0.1 0.2 0.3 0.4)

for noise_frac in ${noise_frac_list[@]}; do

    job_name=N_${dataset}_${tree_type}_${method}_${noise_frac}

    sbatch --cpus-per-task=$ncpu \
           --time=$time \
           --partition=$partition \
           --job-name=$job_name \
           --output=jobs/logs/noise/$job_name \
           --error=jobs/errors/noise/$job_name \
           jobs/noise/runner.sh $dataset $tree_type $method \
           $trunc_frac $update_set $strategy $noise_frac

done
