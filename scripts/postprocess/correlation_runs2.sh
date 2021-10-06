#!/bin/bash

tree_type=$1

dataset_list=('bean' 'compas' 'concrete' 'credit_card' 'energy'
              'german_credit' 'htru2' 'life' 'naval' 'power'
              'spambase' 'surgical' 'wine')

for dataset in ${dataset_list[@]}; do
    python3 scripts/postprocess/correlation.py --dataset $dataset \
                                               --tree_type $tree_type \
                                               --custom_dir 'li' \
                                               --method_list 'random' 'target' 'input_sim' \
                                                             'leaf_sim' 'boostin' 'trex' 'leaf_infSP' \
                                                             'loo' 'subsample' 'leaf_inf' 'leaf_refit'
done
