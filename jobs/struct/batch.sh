
run='jobs/struct/runner.sh'
o='jobs/logs/struct/'
t='lgb'

sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run $t 'loo'
sbatch -a 22    -c 11 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run $t 'loo'

sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out' $run $t 'random'
sbatch -a 22    -c 11 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out' $run $t 'random'
