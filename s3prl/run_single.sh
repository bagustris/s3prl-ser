#!/usr/bin/env bash

# code for batch training emofilm dataset

upstreams="xls_r_300m"
nodes="128"

for upstream in $upstreams; do
    for node in $nodes; do
        for test_fold in fold_1; do
            echo "Training on $test_fold with $upstream using $node nodes"
            python3 run_downstream.py -m train -n emofilm-cv2-$test_fold-$upstream-$node -d emofilm-cv -u $upstream -o "config.downstream_experts.datarc.label_path='./downstream/emofilm-cv2/$test_fold.csv',,config.downstream_experts.datarc.input_dim=$node"
        done
    done
done