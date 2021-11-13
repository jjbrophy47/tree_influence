
run='jobs/inf/runner.sh'
o='jobs/logs/inf/'
t='lgb'

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 1-21  -c 9  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'

sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'

# xgb only
sbatch -a 7,9,14,22 -c 28 -t 4320 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# sgb only
sbatch -a 7,15 -c 28 -t 2000  -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 9    -c 28 -t 10080 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# cb only
sbatch -a 7,11,13-15,18-19 -c 20 -t 1440  -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 12,16            -c 28 -t 1440  -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 17,20-21         -c 20 -t 2880  -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 22               -c 28 -t 5760  -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 6,11,19          -c 28 -t 2880  -p 'long'  -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'
sbatch -a 3                -c 28 -t 10080 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'
sbatch -a 6,11,19          -c 28 -t 2880  -p 'long'  -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit'
sbatch -a 3                -c 28 -t 10080 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit'
sbatch -a 3,6,12,18,21     -c 28 -t 2880  -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 13,16,19         -c 28 -t 4320  -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 1,17             -c 28 -t 10080 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 2,11,20          -c 28 -t 10080 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# scratch pad
sbatch -a 1-21  -c 3  -t 1440 -p 'preempt' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 1-21  -c 3  -t 1440 -p 'preempt' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'
sbatch -a 22    -c 11 -t 1440 -p 'preempt' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 22    -c 11 -t 1440 -p 'preempt' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'
