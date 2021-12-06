
run=jobs/targeted_edit/runner.sh
o=jobs/logs/targeted_edit/
t='lgb'

sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'       $run $t 'random'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'       $run $t 'target'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'     $run $t 'leaf_sim'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'         $run $t 'trex'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'      $run $t 'boostin'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out'   $run $t 'leaf_infSP'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'          $run $t 'loo'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'    $run $t 'subsample'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out'   $run $t 'leaf_inf'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit'

sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinLE-%a.out'    $run $t 'boostinLE'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW1LE-%a.out'  $run $t 'boostinW1LE'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW2LE-%a.out'  $run $t 'boostinW2LE'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSPLE-%a.out' $run $t 'leaf_infSPLE'
sbatch -a 1-22 -c 20 -t 1440 -p 'short' -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# scratch pad

# LGB only
sbatch -a 6                      -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 4                      -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# SGB only
sbatch -a 16                     -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 4                      -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# XGB only
sbatch -a 4                      -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# CB only
sbatch -a 1-3,19-20               -c 20 -t 1440 -p 'short' -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 4                       -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'
