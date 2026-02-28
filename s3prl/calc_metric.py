#!/usr/bin/env python3

import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='Calculate classiffication report. \n Example: ./calc_mosei.py result/downstream/mosei-wavlm/ \n Author: b-atmaja@aist.go.jp', formatter_class=RawTextHelpFormatter)
parser.add_argument('result', type=str, help='prediction directory')
args = parser.parse_args()

# search test prediction and truth files
pred_file = glob.glob(args.result + 'test*predict.txt')[0]
true_file = glob.glob(args.result + 'test*truth.txt')[0]
print(f"Prediction file: {pred_file}")

# read prediction and truth files
pred = pd.read_csv(pred_file, header=None)
true = pd.read_csv(true_file, header=None)

# check to use last column
if pred.shape[1] > 1:
   pred = pred.iloc[:, -1]
if true.shape[1] > 1:
   true = true.iloc[:, -1]

# print(f"Number of samples: {len(pred)}")
acc = np.mean(pred == true)

print(f"Weighted accuracy: {accuracy_score(true, pred)}")
print(f"Unweighted accuracy: {balanced_accuracy_score(true, pred)}")
print(f"Confusion matrix: \n {confusion_matrix(true, pred)} \n")

