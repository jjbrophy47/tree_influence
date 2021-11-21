#!/bin/bash

exp=$1

tree_type_list=('lgb' 'sgb' 'xgb' 'cb')
dataset_list=('adult' 'bank_marketing' 'bean' 'compas' 'concrete' 'credit_card'
              'diabetes' 'energy' 'flight_delays' 'german_credit' 'htru2' 'life'
              'naval' 'no_show' 'obesity' 'power' 'protein' 'spambase'
              'surgical' 'twitter' 'vaccine' 'wine')

for tree_type in ${tree_type_list[@]}; do
    for dataset in ${dataset_list[@]}; do

        # single test
        if [[ $exp = 'remove' ]]; then
            python3 scripts/postprocess/remove.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'label' ]]; then
            python3 scripts/postprocess/label.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'poison' ]]; then
            python3 scripts/postprocess/poison.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'targeted_edit' ]]; then
            python3 scripts/postprocess/targeted_edit.py --tree_type $tree_type --dataset $dataset

        # multi test
        elif [[ $exp = 'remove_set' ]]; then
            python3 scripts/postprocess/remove_set.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'label_set' ]]; then
            python3 scripts/postprocess/label_set.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'poison_set' ]]; then
            python3 scripts/postprocess/poison_set.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'noise_set' ]]; then
            python3 scripts/postprocess/noise_set.py --tree_type $tree_type --dataset $dataset

        # miscellaneous
        elif [[ $exp = 'structure' ]]; then
            python3 scripts/postprocess/structure.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'label_edit' ]]; then
            python3 scripts/postprocess/label_edit.py --tree_type $tree_type --dataset $dataset

        elif [[ $exp = 'counterfactual' ]]; then
            python3 scripts/postprocess/counterfactual.py --tree_type $tree_type --dataset $dataset

        fi

    done
done
