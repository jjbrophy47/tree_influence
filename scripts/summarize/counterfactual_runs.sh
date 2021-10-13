#!/bin/bash

tree_type_list=('lgb' 'xgb' 'sgb' 'cb')

for tree_type in ${tree_type_list[@]}; do
    python3 scripts/summarize/counterfactual.py --tree_type $tree_type
done
