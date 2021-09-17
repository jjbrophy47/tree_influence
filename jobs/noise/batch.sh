
run='jobs/noise/runner.sh'
run2='jobs/noise/runner2.sh'
o='jobs/logs/noise/'
t='lgb'
nf=0.4

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_self-%a.out'       $run $t 'loss'       'self'     $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' 'test_sum' $nf
sbatch -a 1-21  -c 6  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'       'test_sum' $nf
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  'test_sum' $nf
sbatch -a 1,2,3,4,5,6,8,10,11,12,13,16,18,19,20,21 -c 5 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run $t 'loo' 'test_sum' $nf

seed_list=(1 2 3 4 5)

for seed in ${seed_list[@]}; do
    sbatch -a 22-22            -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'      $run2 $t 'trex'      'test_sum'\
        $nf $seed
    sbatch -a 22-22            -c 3  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum'\
        $nf $seed
    sbatch -a 7,9,14,15,17,22  -c 3  -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'       $run2 $t 'loo'       'test_sum'\
        $nf $seed
    sbatch -a 2,3,4,5,10,11,12,13,16,18,19,20,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t\
        'leaf_inf' 'test_sum' $nf $seed
done
