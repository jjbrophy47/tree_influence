#!/bin/bash

tree_type_list=('lgb' 'sgb' 'xgb' 'cb')
dataset_list=('adult' 'bank_marketing' 'bean' 'compas' 'concrete' 'credit_card'
              'diabetes' 'energy' 'flight_delays' 'german_credit' 'htru2' 'life'
              'naval' 'no_show' 'obesity' 'power' 'protein' 'spambase'
              'surgical' 'twitter' 'vaccine' 'wine')

for tree_type in ${tree_type_list[@]}; do
    for dataset in ${dataset_list[@]}; do
        python3 scripts/postprocess/counterfactual.py --tree_type $tree_type --dataset $dataset
    done
done
