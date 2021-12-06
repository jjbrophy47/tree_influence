#!/bin/bash

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb sgb xgb cb

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb

python3 scripts/summarize/correlation.py \
    --tree_type_list sgb

python3 scripts/summarize/correlation.py \
    --tree_type_list xgb

python3 scripts/summarize/correlation.py \
    --tree_type_list cb

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb sgb xgb cb \
    --in_sub_dir li \
    --out_sub_dir li

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb \
    --in_sub_dir li \
    --out_sub_dir li

python3 scripts/summarize/correlation.py \
    --tree_type_list sgb \
    --in_sub_dir li \
    --out_sub_dir li

python3 scripts/summarize/correlation.py \
    --tree_type_list xgb \
    --in_sub_dir li \
    --out_sub_dir li

python3 scripts/summarize/correlation.py \
    --tree_type_list cb \
    --in_sub_dir li \
    --out_sub_dir li

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb sgb xgb cb \
    --out_sub_dir regression \
    --dataset_list concrete energy life naval obesity power protein wine

python3 scripts/summarize/correlation.py \
    --tree_type_list lgb sgb xgb cb \
    --out_sub_dir classification \
    --dataset_list adult bank_marketing bean compas credit_card diabetes \
        flight_delays german_credit htru2 no_show spambase surgical twitter vaccine
