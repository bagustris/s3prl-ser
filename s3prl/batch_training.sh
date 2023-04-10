#!/usr/bin/env bash

# code for batch training emofilm dataset

upstreams="xls_r_300m xls_r_1b xls_r_2b unispeech_sat_large wavlm_large"

for upstream in $upstreams; do
    for test_fold in fold_1.csv fold_2.csv fold_3.csv fold_4.csv fold_5.csv; do
        echo "Training on $test_fold with $upstream"
        python3 run_downstream.py -m train -n emofilm-cv-$upstream-256 -d emofilm-cv -u $upstream -o "config.downstream_experts.datarc.label_path='./downstream/emofilm-cv/$test_fold'"
    done
done
