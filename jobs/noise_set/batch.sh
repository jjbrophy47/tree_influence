
run='jobs/noise_set/runner.sh'
o='jobs/logs/noise_set/'
t='lgb'
nf=0.4
st='test_sum'
ss='self'

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_self-%a.out'       $run $t 'loss'       $ss $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    $ss $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' $st $nf
sbatch -a 1-21  -c 9  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'       $st $nf
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  $st $nf
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        $st $nf
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' $st $nf
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'   $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'  $st $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'  $st $nf

sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_loss-%a.out'       $run $t 'loss'       $ss $nf
sbatch -a 22 -c 11 -t 2880 -p 'long'  -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    $ss $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' $st $nf
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'       $st $nf
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  $st $nf
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'  $run $t 'boostinW1'  $st $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'  $run $t 'boostinW2'  $st $nf

# lgb only
sbatch -a 22 -c 28 -t 4320  -p 'long'   -o ${o}${t}'_subsample-%a.out' $run $t 'subsample'  $st $nf
sbatch -a 6  -c 20 -t 1440  -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'   $st $nf

# xgb only
sbatch -a 7,9 -c 28 -t 2880 -p 'long'   -o ${o}${t}'_loo-%a.out'      $run $t 'loo' $st $nf
sbatch -a 22  -c 28 -t 4320 -p 'long'   -o ${o}${t}'_loo-%a.out'      $run $t 'loo' $st $nf

# sgb only
sbatch -a 7  -c 28 -t 2880  -p 'long'   -o ${o}${t}'_loo-%a.out'       $run $t 'loo'        $st $nf
sbatch -a 9  -c 28 -t 7200  -p 'long'   -o ${o}${t}'_loo-%a.out'       $run $t 'loo'        $st $nf
sbatch -a 22 -c 28 -t 4320  -p 'long'   -o ${o}${t}'_subsample-%a.out' $run $t 'subsample'  $st $nf

# cb only
sbatch -a 1-21             -c 28 -t 1440  -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  $st $nf
sbatch -a 22               -c 28 -t 5760  -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  $st $nf
sbatch -a 11               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'   $st $nf
sbatch -a 11               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' $st $nf
sbatch -a 6,18,21          -c 28 -t 1440  -p 'short' -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        $st $nf
sbatch -a 3,13,15,19       -c 28 -t 2880  -p 'long'  -o ${o}${t}'_loo-%a.out'        $run $t 'loo'        $st $nf

# scratch pad
sbatch -a 11,13               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'   $st $nf
sbatch -a 11,13               -c 28 -t 1440  -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' $st $nf
