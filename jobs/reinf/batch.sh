
run='jobs/reinf/runner.sh'
o='jobs/logs/reinf/'
t='lgb'

sbatch -a 4-5,8,10,12,21 -c 20 -t 1440 -p 'short' -o ${o}${t}'_random_r-%a.out'     $run $t 'random'     'reestimate' 0
sbatch -a 4-5,8,10,12,21 -c 15 -t 1440 -p 'short' -o ${o}${t}'_boostin_r-%a.out'    $run $t 'boostin'    'reestimate' 0
sbatch -a 4-5,8,10,12,21 -c 15 -t 1440 -p 'short' -o ${o}${t}'_loo_r-%a.out'        $run $t 'loo'        'reestimate' 0
sbatch -a 4-5,8,10,12,21 -c 15 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit_r-%a.out' $run $t 'leaf_refit' 'reestimate' 0

sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_random_f-%a.out'     $run $t 'random'      'fixed' 0
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_boostin_f-%a.out'    $run $t 'boostin'     'fixed' 0
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_loo_f-%a.out'        $run $t 'loo'         'fixed' 0
sbatch -a 4-5,8,10,12,21 -c 7 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit_f-%a.out' $run $t 'leaf_refit'  'fixed' 0

# scratch pad
sbatch -a 5,8,10 -c 40 -t 1440 -p 'preempt' -o ${o}${t}'_loo_r-%a.out'        $run $t 'loo'        'reestimate' 0
sbatch -a 21     -c 40 -t 1440 -p 'preempt' -o ${o}${t}'_loo_f-%a.out'        $run $t 'loo'        'fixed'      0
sbatch -a 21     -c 40 -t 1440 -p 'preempt' -o ${o}${t}'_leaf_refit_f-%a.out' $run $t 'leaf_refit' 'fixed'      0
