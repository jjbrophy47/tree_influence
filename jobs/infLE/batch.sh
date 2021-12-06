
run='jobs/infLE/runner.sh'
o='jobs/logs/infLE/'
t='lgb'

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinLE-%a.out'    $run $t 'boostinLE'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSPLE-%a.out' $run $t 'leaf_infSPLE'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinLEW1-%a.out'  $run $t 'boostinLEW1'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostinLEW2-%a.out'  $run $t 'boostinLEW2'

sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinLE-%a.out'    $run $t 'boostinLE'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSPLE-%a.out' $run $t 'leaf_infSPLE'
sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinLEW1-%a.out'  $run $t 'boostinLEW1'
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostinLEW2-%a.out'  $run $t 'boostinLEW2'

# lgb only
sbatch -a 22                    -c 28 -t 2000  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 3,6,11,13,19          -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 3,6,11,13,19          -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# sgb only
sbatch -a 7,15,22               -c 28 -t 2880  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 9                     -c 28 -t 10080 -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 3,13                  -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 3,13                  -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# xgb only
sbatch -a 7,9,14,22             -c 28 -t 4320  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 3,11                  -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 3,11                  -c 11 -t 2880  -p 'long'  -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'

# cb only
sbatch -a 3,6,11,13        -c 28 -t 4320  -p 'long'  -o ${o}${t}'_leaf_infLE-%a.out'   $run $t 'leaf_infLE'
sbatch -a 3,6,11,13        -c 28 -t 4320  -p 'long'  -o ${o}${t}'_leaf_refitLE-%a.out' $run $t 'leaf_refitLE'
sbatch -a 12,18            -c 28 -t 2880  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 6,11,13,16,19,21 -c 15 -t 4320  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
sbatch -a 1-3,11,17,20     -c 28 -t 10080 -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'

# scratch pad
sbatch -a 6,11,13,16,19,21 -c 15 -t 4320  -p 'long'  -o ${o}${t}'_looLE-%a.out'        $run $t 'looLE'
