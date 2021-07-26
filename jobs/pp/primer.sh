dataset=$1
model=$2
tune_frac=$3
mem=$4
time=$5
partition=$6

job_name=PP_${dataset}_${model}

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/pp/$job_name \
       --error=jobs/errors/pp/$job_name \
       jobs/pp/runner.sh $dataset $model $tune_frac
