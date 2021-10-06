#!/bin/bash

tree_type=$1

dataset_list=('adult' 'bank_marketing' 'bean' 'compas' 'concrete'
              'credit_card' 'diabetes' 'energy' 'flight_delays'
              'german_credit' 'htru2' 'life' 'naval' 'no_show'
              'obesity' 'power' 'protein' 'spambase' 'surgical'
              'twitter' 'vaccine' 'wine')

for dataset in ${dataset_list[@]}; do
    python3 scripts/postprocess/correlation.py --dataset $dataset --tree_type $tree_type
done
