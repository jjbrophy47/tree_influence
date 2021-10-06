
run='jobs/noise/runner.sh'
run2='jobs/noise/runner2.sh'
o='jobs/logs/noise/'
t='lgb'
nf=0.4
ts='test_sum'
sl='self'
seed_list=(1 2 3 4 5)

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_self-%a.out'       $run $t 'loss'       'self'     $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    'test_sum' $nf
sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' 'test_sum' $nf
sbatch -a 1-21  -c 6  -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'       'test_sum' $nf
sbatch -a 1-21  -c 5  -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  'test_sum' $nf
sbatch -a 1,2,3,4,5,6,8,10,11,12,13,16,18,19,20,21 -c 5 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run $t 'loo' 'test_sum' $nf

sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_self-%a.out'       $run $t 'loss'       'self'     $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' 'test_sum' $nf

for seed in ${seed_list[@]}; do
    sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out' $run2 $t 'trex' 'test_sum' $nf $seed
    sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 7,9,14,15,17,22 -c 3 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t\
        'leaf_inf' 'test_sum' $nf $seed
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t\
        'leaf_refit' 'test_sum' $nf $seed
done

# lgb only
for seed in ${seed_list[@]}; do
    sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 22 -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
    sbatch -a 6  -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t 'leaf_inf' 'test_sum' $nf $seed
    sbatch -a 19 -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t 'leaf_refit' 'test_sum' $nf $seed
done

# xgb only
for seed in ${seed_list[@]}; do
    sbatch -a 22   -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 7    -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
    sbatch -a 9,22 -c 28 -t 4320 -p 'long'  -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
done

# sgb only
for seed in ${seed_list[@]}; do
    sbatch -a 22   -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 7    -c 28 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
    sbatch -a 9,22 -c 28 -t 4320 -p 'long'  -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
done

# cb only
for seed in ${seed_list[@]}; do
    sbatch -a 1-6,8 -c 15 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum'  $nf $seed
    sbatch -a 7,9-11,13-15,18-19 -c 20 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 12,16 -c 28 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample' 'test_sum' $nf $seed 
    sbatch -a 17,20-21 -c 20 -t 2880 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 22 -c 28 -t 5760 -p 'long'  -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 3-6,8,10-13,16,18-21 -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run $t 'leaf_refit' 'test_sum' $nf $seed
    sbatch -a 3-6,8,10-13,16,18-21 -c 28 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run $t 'leaf_inf' 'test_sum' $nf $seed
done

# scratch pad
for seed in ${seed_list[@]}; do
    # sbatch -a 7,9,14,22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' 'test_sum' $nf $seed
    sbatch -a 1,7,9,14,15,22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out' $run2 $t 'loo' 'test_sum' $nf $seed
    # sbatch -a 3,6,8 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t 'leaf_inf' 'test_sum' $nf $seed
    # sbatch -a 3-6,8,10-13,16,18-19,21 -c 11 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t 'leaf_refit' 'test_sum' $nf $seed
done

sbatch -a 1-21  -c 3  -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     'test_sum' $nf
sbatch -a 22 -c 11 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     'test_sum' $nf

sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_loss-%a.out'       $run $t 'loss'       $sl $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'     $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'     $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'  $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'   $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'    $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP' $ts $nf
sbatch -a 5,8,12,13,14-17,21    -c 15 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'       $ts $nf
sbatch -a 5,8,12,13,15-17,21-22 -c 15 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out'  $run $t 'subsample'  $ts $nf
