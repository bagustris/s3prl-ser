#!/usr/bin/env python3

import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from sklearn.metrics import balanced_accuracy_score

parser = argparse.ArgumentParser(
    description='Calculate weighted accuracy of mosei. \n Example: ./calc_mosei.py result/downstream/mosei-wavlm/ \n Author: b-atmaja@aist.go.jp', formatter_class=RawTextHelpFormatter)
parser.add_argument('result', type=str, help='prediction directory')
args = parser.parse_args()

pred_file = args.result + 'test_predict.txt'
true_file = args.result + 'test_truth.txt'

pred = np.loadtxt(pred_file)
true = np.loadtxt(true_file)
acc = np.mean(pred == true)
bacc = balanced_accuracy_score(true, pred)
print(f"Weighted accuracy: {acc*100:.2f}")
print(f"Balanced accuracy: {bacc*100:.2f}")
