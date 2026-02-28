#!/usr/bin/env python3
"""
Language-specific Accuracy Calculator for EmoFilm Dataset

This script calculates weighted and unweighted accuracy metrics for each language
(English, Italian, Spanish) in the EmoFilm dataset results.

Usage:
    python calc_metric_lang.py <result_directory> [--label-csv <path_to_csv>]

Example:
    python calc_metric_lang.py result/downstream/emofilm-wavlm/

The script will:
1. Load prediction and truth files from the result directory
2. Map predictions to language information from the EmoFilm CSV file
3. Calculate overall and language-specific metrics
4. Save detailed results to a text file

Output metrics:
- Weighted accuracy (standard accuracy)
- Unweighted accuracy (balanced accuracy for class imbalance)
- Confusion matrices for overall and per-language results
"""

import argparse
from argparse import RawTextHelpFormatter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Calculate language-specific classification metrics for EmoFilm dataset.\n"
        "Example: python calc_metric_lang.py result/downstream/emofilm-wavlm/ --label-csv downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv\n"
        "Author: bagus.tris@naist.ac.jp",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("result", type=str, help="prediction directory")
    parser.add_argument(
        "--label-csv",
        type=str,
        default="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv",
        help="path to the EmoFilm label CSV file",
    )
    args = parser.parse_args()

    # Search test prediction and truth files
    pred_files = glob.glob(args.result + "test*predict.txt")
    true_files = glob.glob(args.result + "test*truth.txt")

    if not pred_files or not true_files:
        print("Error: Could not find prediction or truth files in the result directory")
        print(f"Looking for files in: {args.result}")
        return

    pred_file = pred_files[0]
    true_file = true_files[0]
    print(f"Prediction file: {pred_file}")
    print(f"Truth file: {true_file}")

    # Read prediction and truth files
    pred = pd.read_csv(pred_file, header=None)
    true = pd.read_csv(true_file, header=None)

    # Check to use last column
    if pred.shape[1] > 1:
        pred = pred.iloc[:, -1]
    if true.shape[1] > 1:
        true = true.iloc[:, -1]

    # Load the label CSV file to get language information
    try:
        df_labels = pd.read_csv(args.label_csv)
        print(f"Loaded label file: {args.label_csv}")
    except FileNotFoundError:
        print(f"Error: Label file not found: {args.label_csv}")
        return

    # Filter for test set only (set == 1 in the CSV)
    test_df = df_labels[df_labels["set"] == 1].reset_index(drop=True)

    if len(test_df) != len(pred):
        print("Warning: Mismatch in number of samples!")
        print(f"Test samples in CSV: {len(test_df)}")
        print(f"Prediction samples: {len(pred)}")
        print("Using minimum length for analysis...")
        min_len = min(len(test_df), len(pred))
        test_df = test_df.iloc[:min_len]
        pred = pred.iloc[:min_len]
        true = true.iloc[:min_len]

    # Add predictions and truth to the dataframe
    test_df = test_df.copy()
    test_df["predicted"] = pred.values
    test_df["true_label"] = true.values

    # Calculate overall metrics
    overall_acc = accuracy_score(test_df["true_label"], test_df["predicted"])
    overall_balanced_acc = balanced_accuracy_score(
        test_df["true_label"], test_df["predicted"]
    )

    print("\n=== OVERALL RESULTS ===")
    print(f"Total samples: {len(test_df)}")
    print(f"Weighted accuracy: {overall_acc:.4f}")
    print(f"Unweighted accuracy (balanced): {overall_balanced_acc:.4f}")
    print(
        f"Confusion matrix:\n{confusion_matrix(test_df['true_label'], test_df['predicted'])}"
    )

    # Calculate language-specific metrics
    languages = ["en", "it", "es"]

    print("\n=== LANGUAGE-SPECIFIC RESULTS ===")
    lang_results = {}

    for lang in languages:
        lang_data = test_df[test_df["language"] == lang]

        if len(lang_data) == 0:
            print(f"\n{lang.upper()}: No samples found")
            continue

        lang_pred = lang_data["predicted"]
        lang_true = lang_data["true_label"]

        lang_acc = accuracy_score(lang_true, lang_pred)
        lang_balanced_acc = balanced_accuracy_score(lang_true, lang_pred)
        lang_cm = confusion_matrix(lang_true, lang_pred)

        lang_results[lang] = {
            "samples": len(lang_data),
            "weighted_acc": lang_acc,
            "unweighted_acc": lang_balanced_acc,
            "confusion_matrix": lang_cm,
        }

        print(f"\n{lang.upper()}:")
        print(f"  Samples: {len(lang_data)}")
        print(f"  Weighted accuracy: {lang_acc:.4f}")
        print(f"  Unweighted accuracy (balanced): {lang_balanced_acc:.4f}")
        print(f"  Confusion matrix:\n{lang_cm}")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(
        f"{'Language':<10} {'Samples':<8} {'Weighted Acc':<14} {'Unweighted Acc':<16}"
    )
    print(f"{'-'*50}")
    print(
        f"{'Overall':<10} {len(test_df):<8} {overall_acc:<14.4f} {overall_balanced_acc:<16.4f}"
    )

    for lang in languages:
        if lang in lang_results:
            result = lang_results[lang]
            print(
                f"{lang.upper():<10} {result['samples']:<8} {result['weighted_acc']:<14.4f} {result['unweighted_acc']:<16.4f}"
            )

    # Save detailed results to file
    output_file = args.result.rstrip("/") + "_language_metrics.txt"
    with open(output_file, "w") as f:
        f.write("=== LANGUAGE-SPECIFIC EMOFILM RESULTS ===\n\n")
        f.write(f"Prediction file: {pred_file}\n")
        f.write(f"Truth file: {true_file}\n")
        f.write(f"Label file: {args.label_csv}\n\n")

        f.write("=== OVERALL RESULTS ===\n")
        f.write(f"Total samples: {len(test_df)}\n")
        f.write(f"Weighted accuracy: {overall_acc:.4f}\n")
        f.write(f"Unweighted accuracy (balanced): {overall_balanced_acc:.4f}\n")
        f.write(
            f"Confusion matrix:\n{confusion_matrix(test_df['true_label'], test_df['predicted'])}\n\n"
        )

        f.write("=== LANGUAGE-SPECIFIC RESULTS ===\n")
        for lang in languages:
            if lang in lang_results:
                result = lang_results[lang]
                f.write(f"\n{lang.upper()}:\n")
                f.write(f"  Samples: {result['samples']}\n")
                f.write(f"  Weighted accuracy: {result['weighted_acc']:.4f}\n")
                f.write(
                    f"  Unweighted accuracy (balanced): {result['unweighted_acc']:.4f}\n"
                )
                f.write(f"  Confusion matrix:\n{result['confusion_matrix']}\n")

        f.write("\n=== SUMMARY TABLE ===\n")
        f.write(
            f"{'Language':<10} {'Samples':<8} {'Weighted Acc':<14} {'Unweighted Acc':<16}\n"
        )
        f.write(f"{'-'*50}\n")
        f.write(
            f"{'Overall':<10} {len(test_df):<8} {overall_acc:<14.4f} {overall_balanced_acc:<16.4f}\n"
        )

        for lang in languages:
            if lang in lang_results:
                result = lang_results[lang]
                f.write(
                    f"{lang.upper():<10} {result['samples']:<8} {result['weighted_acc']:<14.4f} {result['unweighted_acc']:<16.4f}\n"
                )

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
