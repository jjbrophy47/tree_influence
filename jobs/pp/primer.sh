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
       --output=jobs/logs/pp/$job_name \
       --error=jobs/errors/pp/$job_name \
       jobs/pp/runner.sh $dataset $model
