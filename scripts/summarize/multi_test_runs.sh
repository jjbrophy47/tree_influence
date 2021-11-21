#!/bin/bash

exp=$1

tree_type_list=('lgb' 'sgb' 'xgb' 'cb')
ckpt_list=(1 2 3 4 5 6 7 8 9 10)

if [[ $exp = 'noise_set' ]]; then
    ckpt_list=(1 2 3 4 5 6)
fi

for tree_type in ${tree_type_list[@]}; do
    for ckpt in ${ckpt_list[@]}; do

        if [[ $exp = 'remove_set' ]]; then
            python3 scripts/summarize/remove_set.py --tree_type $tree_type --ckpt $ckpt

        elif [[ $exp = 'label_set' ]]; then
            python3 scripts/summarize/label_set.py --tree_type $tree_type --ckpt $ckpt

        elif [[ $exp = 'poison_set' ]]; then
            python3 scripts/summarize/poison_set.py --tree_type $tree_type --ckpt $ckpt

        elif [[ $exp = 'noise_set' ]]; then
            python3 scripts/summarize/noise_set.py --tree_type $tree_type --ckpt $ckpt

        fi

    done
done
