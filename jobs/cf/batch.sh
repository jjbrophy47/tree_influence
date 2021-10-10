run='jobs/cf/runner.sh'
o='jobs/logs/cf/'
t='lgb'
s=10

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t $s 'random'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t $s 'target'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t $s 'input_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t $s 'leaf_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t $s 'boostin'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t $s 'leaf_infSP'
sbatch -a 1-21  -c 9  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t $s 'trex'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t $s 'loo'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t $s 'leaf_inf'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t $s 'leaf_refit'

sbatch -a 22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t $s 'random'
sbatch -a 22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t $s 'target'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t $s 'input_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t $s 'leaf_sim'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t $s 'boostin'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t $s 'leaf_infSP'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t $s 'trex'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t $s 'loo'

# xgb only
sbatch -a 7,9,22  -c 20  -t 4320 -p 'long'  -o ${o}${t}'_random-%a.out'     $run $t $s 'random'
sbatch -a 7,9,22  -c 20  -t 4320 -p 'long'  -o ${o}${t}'_target-%a.out'     $run $t $s 'target'
sbatch -a 22      -c 28  -t 2880 -p 'long'  -o ${o}${t}'_trex-%a.out'       $run $t $s 'trex'
sbatch -a 22      -c 28  -t 2880 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'

# sgb only
sbatch -a 7,9,14,15,22 -c 20 -t 4320 -p 'long'  -o ${o}${t}'_random-%a.out'     $run $t $s 'random'
sbatch -a 7,9,14,22    -c 20 -t 4320 -p 'long'  -o ${o}${t}'_target-%a.out'     $run $t $s 'target'
sbatch -a 22           -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t $s 'leaf_infSP'
sbatch -a 22           -c 28 -t 2880 -p 'long'  -o ${o}${t}'_trex-%a.out'       $run $t $s 'trex'
sbatch -a 7,9,15,22    -c 28 -t 2880 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 22           -c 28 -t 2880 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t $s 'loo'

# cb only
sbatch -a 7,11,13-15,18-19     -c 20 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 12,16                -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 17,20-21             -c 20 -t 2880 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 22                   -c 28 -t 5760 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t $s 'subsample'
sbatch -a 3-6,8,10-13,16,18-21 -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t $s 'leaf_inf'
sbatch -a 3-6,8,10-13,16,18-21 -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t $s 'leaf_refit'
sbatch -a 3,6,12,18,21         -c 28 -t 2880 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t $s 'loo'
sbatch -a 13,16,19             -c 28 -t 4320 -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t $s 'loo'

# scratch pad
sbatch -a 6,8   -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t $s 'leaf_inf'
