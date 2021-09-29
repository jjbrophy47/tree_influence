
run='jobs/reinf/runner.sh'
o='jobs/logs/reinf/'
t='lgb'

sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     'reestimate'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    'reestimate'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        'reestimate'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' 'reestimate'

sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     'fixed'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    'fixed'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        'fixed'
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' 'fixed'
