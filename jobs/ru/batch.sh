
run='jobs/ru/runner.sh'
run2='jobs/ru/runner2.sh'
o='jobs/logs/ru/'
t='xgb'

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 1-21  -c 6  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'
sbatch -a 1,2,3,4,5,6,8,10,11,12,13,16,18,19,20,21 -c 5 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run $t 'loo'

seed_list=(1 2 3 4 5)

for seed in ${seed_list[@]}; do
    sbatch -a 22-22            -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'      $run2 $t 'trex'      $seed
    sbatch -a 22-22            -c 3  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' $seed
    sbatch -a 7,9,14,15,17,22  -c 3  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'       $run2 $t 'loo'       $seed
    sbatch -a 2,3,4,5,10,11,12,13,16,18,19,20,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t 'leaf_inf' $seed
done
