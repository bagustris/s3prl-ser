#!/usr/bin/env python3

# script to calculate performance
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import argparse
import glob
import pandas as pd


def get_classification_report(folder_path):

    glob_pred = glob.glob(os.path.join(folder_path, "test*predict.txt"))
    glob_truth = glob.glob(os.path.join(folder_path, "test*truth.txt"))
    print(f"find prediction file: {glob_pred}")
    print(f"find truth file: {glob_truth}")

    pred_file = os.path.join(glob_pred[0])
    truth_file = os.path.join(glob_truth[0])

    # read the files using pandas, separator is space
    # if it contains a single column, then use that column
    # it contains two columns, then use the second column
    pred_df = pd.read_csv(pred_file, sep="\s+", header=None)
    truth_df = pd.read_csv(truth_file, sep="\s+", header=None)

    # print shape and length
    print(f"Prediction columns: {pred_df.shape[1]}")
    print(f"Truth columns: {truth_df.shape[1]}")

    # print columns
    if pred_df.shape[1] == 1:
        pred_data = pred_df.iloc[:, 0]
        true_data = truth_df.iloc[:, 0]
        # print unique values
        print(f"labels in prediction: {pred_data.unique()}")

    elif pred_df.shape[1] == 2:
        pred_data = pred_df.iloc[:, 1]
        true_data = truth_df.iloc[:, 1]
        # print unique values
        print(f"labels in prediction: {pred_data.unique()}")

    report = classification_report(true_data, pred_data)
    print(report)

    accuracy = accuracy_score(true_data, pred_data)
    balanced_accuracy = balanced_accuracy_score(true_data, pred_data)
    f1 = f1_score(true_data, pred_data, average="weighted")

    print(f"WA: {accuracy*100:.2f}")
    print(f"UA: {balanced_accuracy*100:.2f}")
    print(f"F1 Score: {f1*100:.2f}")


# Usage
# ./calc_perf.py folder_path

parser = argparse.ArgumentParser()
parser.add_argument(
    'folder_path',
    help='Path to the folder',
    type=str)
# required=True)
args = parser.parse_args()

get_classification_report(args.folder_path)
