
run=jobs/targeted_edit/runner.sh
o=jobs/logs/targeted_edit/
t='lgb'

sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'       $run $t 'random'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'     $run $t 'leaf_sim'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'         $run $t 'trex'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'      $run $t 'boostin'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW1-%a.out'    $run $t 'boostinW1'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinW2-%a.out'    $run $t 'boostinW2'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinLE-%a.out'    $run $t 'boostinLE'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinLEW1-%a.out'  $run $t 'boostinLEW1'
sbatch -a 1-22  -c 20 -t 1440 -p 'short' -o ${o}${t}'_boostinLEW2-%a.out'  $run $t 'boostinLEW2'

# LGB only
sbatch -a 7,9,17,22          -c 1 --mem=60GB -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 15                 -c 1 --mem=60GB -t 2880 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'

# SGB only
sbatch -a 7,17,22            -c 1 --mem=60GB -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 9                  -c 1 --mem=60GB -t 2880 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 15                 -c 1 --mem=60GB -t 4320 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'

# XGB only
sbatch -a 9,17,22            -c 1 --mem=60GB -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 15                 -c 1 --mem=60GB -t 4320 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'

# CB only
sbatch -a 2,17               -c 1 --mem=60GB -t 2880 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 7,9,14-15          -c 1 --mem=60GB -t 4320 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'
sbatch -a 22                 -c 1 --mem=60GB -t 7200 -p 'long'  -o ${o}${t}'_trex-%a.out' $run $t 'trex'

# scratch pad
