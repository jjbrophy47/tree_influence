
run='jobs/ru/runner.sh'
run2='jobs/ru/runner2.sh'
o='jobs/logs/ru/'
t='lgb'
seed_list=(1 2 3 4 5)

sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 3-6,8,10-13,16,18-19,21 -c 6 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'

for seed in ${seed_list[@]}; do
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' $seed
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_loo-%a.out'      $run2 $t 'loo'      $seed
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_inf-%a.out' $run2 $t 'leaf_inf' $seed
    sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t 'leaf_refit' $seed
done

# cb only
for seed in ${seed_list[@]}; do
    # sbatch -a 3-6,8,10-13,16,18-19,21 -c 3 -t 4320 -p 'long' -o ${o}${t}'_subsample-%a.out' $run2 $t 'subsample' $seed
    sbatch -a 3 -c 3 -t 4320 -p 'long' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t 'leaf_refit' $seed
    sbatch -a 3 -c 3 -t 4320 -p 'long' -o ${o}${t}'_leaf_inf-%a.out'   $run2 $t 'leaf_inf' $seed
done

# scratch pad
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_random-%a.out'     $run $t 'random'
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_target-%a.out'     $run $t 'target'
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_input_sim-%a.out'  $run $t 'input_sim'
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_sim-%a.out'   $run $t 'leaf_sim'
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_boostin-%a.out'    $run $t 'boostin'
sbatch -a 3 -c 3 -t 1440 -p 'short' -o ${o}${t}'_leaf_infSP-%a.out' $run $t 'leaf_infSP'
sbatch -a 3 -c 6 -t 1440 -p 'short' -o ${o}${t}'_trex-%a.out'       $run $t 'trex'

for seed in ${seed_list[@]}; do
    sbatch -a 3 -c 3 -t 2880 -p 'long' -o ${o}${t}'_leaf_refit-%a.out' $run2 $t 'leaf_refit' $seed
done
