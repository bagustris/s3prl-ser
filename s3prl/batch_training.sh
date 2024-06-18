#!/usr/bin/env bash

# code for batch training emofilm dataset

upstreams="xls_r_300m xls_r_1b xls_r_2b unispeech_sat_large wavlm_large"

#upstreams="unispeech_sat_large wavlm_large"
nodes="128"

for upstream in $upstreams; do
    for test_fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
        echo "Training on $test_fold with $upstream"
        python3 run_downstream.py -m train -n emofilm-cv-$test_fold-$upstream-$nodes -d emofilm-cv -u $upstream -o "config.downstream_experts.datarc.label_path='./downstream/emofilm-cv/$test_fold.csv',,config.downstream_experts.datarc.input_dim=$nodes"
    done
done
