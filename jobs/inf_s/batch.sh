
run='jobs/inf_s/runner.sh'
o='jobs/logs/inf_s/'
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
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'

sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'

# xgb only
sbatch -a 7,9 -c 28 -t 2880 -p 'long'   -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 22  -c 28 -t 4320 -p 'long'   -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# sgb only
sbatch -a 7 -c 28 -t 2880  -p 'long'   -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 9 -c 28 -t 7200  -p 'long'   -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# cb only
sbatch -a 1-21             -c 28 -t 1440  -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 22               -c 28 -t 5760  -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 11               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'
sbatch -a 11               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit'
sbatch -a 6,18,21          -c 28 -t 1440  -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'
sbatch -a 3,13,15,19       -c 28 -t 2880  -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'

# scratch pad
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'
sbatch -a 22    -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'
sbatch -a 22    -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'
