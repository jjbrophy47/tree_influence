dataset=$1
model=$2
mem=$3
time=$4
partition=$5

job_name=PP_${dataset}_${model}

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/predictive_performance/$job_name \
       --error=jobs/errors/predictive_performance/$job_name \
       jobs/predictive_performance/runner.sh $dataset $model
