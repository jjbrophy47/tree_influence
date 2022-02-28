#!/bin/bash

exp=$1

python3 scripts/rank/${exp}.py --tree_type 'lgb'
python3 scripts/rank/${exp}.py --tree_type 'sgb'
python3 scripts/rank/${exp}.py --tree_type 'xgb'
python3 scripts/rank/${exp}.py --tree_type 'cb'
python3 scripts/rank/${exp}.py --tree_type 'lgb' 'sgb' 'xgb'
python3 scripts/rank/${exp}.py --tree_type 'lgb' 'sgb' 'xgb' 'cb'
