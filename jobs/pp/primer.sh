dataset=$1
model=$2
tune_frac=$3
train_frac=$4
mem=$5
time=$6
partition=$7

job_name=PP_${dataset}_${model}

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/pp/$job_name \
       --error=jobs/errors/pp/$job_name \
       jobs/pp/runner.sh $dataset $model $tune_frac $train_frac
