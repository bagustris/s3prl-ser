#!/usr/bin/env python3
"""
Confusion Matrix Plotter for EmoFilm Dataset

This script creates confusion matrix plots with real emotion labels for EmoFilm dataset results.
It generates both overall and language-specific confusion matrices with proper emotion names.

Usage:
    python plot_cm.py <result_directory> [--label-csv <path_to_csv>] [--save-dir <output_directory>]

Example:
    python plot_cm.py result/downstream/emofilm-wavlm/

The script will:
1. Load prediction and truth files from the result directory
2. Map predictions to language information from the EmoFilm CSV file
3. Generate confusion matrices with real emotion labels
4. Save plots as PNG files for overall and per-language results
"""

import argparse
from argparse import RawTextHelpFormatter
from sklearn.metrics import confusion_matrix
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Emotion label mapping based on EmoFilm dataset
# Inferred from dataset.py comments and typical emotion recognition datasets
EMOTION_LABELS = {
    0: "Anger",
    1: "Contempt",
    2: "Happy",
    3: "Neutral",  # Assuming neutral based on typical 5-class emotion recognition
    4: "Sad",
}


def plot_confusion_matrix(cm, labels, title, save_path=None, figsize=(8, 6)):
    """
    Plot a confusion matrix with proper formatting

    Args:
        cm: confusion matrix array
        labels: list of emotion labels
        title: plot title
        save_path: path to save the plot (optional)
        figsize: figure size tuple
    """
    plt.figure(figsize=figsize)

    # Calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    # Add percentage annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                plt.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm_percent[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="red",
                )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.ylabel("True Emotion", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices with real emotion labels for EmoFilm dataset.\n"
        "Example: python plot_cm.py result/downstream/emofilm-wavlm/\n"
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
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="directory to save plots (default: same as result directory)",
    )
    args = parser.parse_args()

    # Set save directory
    if args.save_dir is None:
        args.save_dir = args.result.rstrip("/")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

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

    # Get unique emotion labels present in the data
    all_emotions = sorted(
        set(test_df["true_label"].tolist() + test_df["predicted"].tolist())
    )
    emotion_names = [EMOTION_LABELS.get(emo, f"Emotion_{emo}") for emo in all_emotions]

    print(f"\nEmotion classes found: {all_emotions}")
    print(f"Emotion names: {emotion_names}")

    # Plot overall confusion matrix
    overall_cm = confusion_matrix(
        test_df["true_label"], test_df["predicted"], labels=all_emotions
    )

    print("\n=== OVERALL CONFUSION MATRIX ===")
    overall_save_path = os.path.join(args.save_dir, "confusion_matrix_overall.png")
    plot_confusion_matrix(
        overall_cm,
        emotion_names,
        f"Overall Confusion Matrix (n={len(test_df)})",
        save_path=overall_save_path,
    )

    # Plot language-specific confusion matrices
    languages = ["en", "it", "es"]

    print("\n=== LANGUAGE-SPECIFIC CONFUSION MATRICES ===")

    for lang in languages:
        lang_data = test_df[test_df["language"] == lang]

        if len(lang_data) == 0:
            print(f"\n{lang.upper()}: No samples found")
            continue

        lang_pred = lang_data["predicted"]
        lang_true = lang_data["true_label"]

        # Get emotions present in this language subset
        lang_emotions = sorted(set(lang_true.tolist() + lang_pred.tolist()))
        lang_emotion_names = [
            EMOTION_LABELS.get(emo, f"Emotion_{emo}") for emo in lang_emotions
        ]

        lang_cm = confusion_matrix(lang_true, lang_pred, labels=lang_emotions)

        print(f"\n{lang.upper()}: {len(lang_data)} samples")
        lang_save_path = os.path.join(args.save_dir, f"confusion_matrix_{lang}.png")
        plot_confusion_matrix(
            lang_cm,
            lang_emotion_names,
            f"{lang.upper()} Confusion Matrix (n={len(lang_data)})",
            save_path=lang_save_path,
        )

    # Create a summary figure with all language CMs in subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "EmoFilm Confusion Matrices by Language", fontsize=16, fontweight="bold"
    )

    # Overall CM
    sns.heatmap(
        overall_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        ax=axes[0, 0],
        cbar=False,
    )
    axes[0, 0].set_title(f"Overall (n={len(test_df)})")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")

    # Language-specific CMs
    for idx, lang in enumerate(languages):
        row = (idx + 1) // 2
        col = (idx + 1) % 2

        lang_data = test_df[test_df["language"] == lang]

        if len(lang_data) > 0:
            lang_pred = lang_data["predicted"]
            lang_true = lang_data["true_label"]
            lang_emotions = sorted(set(lang_true.tolist() + lang_pred.tolist()))
            lang_emotion_names = [
                EMOTION_LABELS.get(emo, f"Emotion_{emo}") for emo in lang_emotions
            ]
            lang_cm = confusion_matrix(lang_true, lang_pred, labels=lang_emotions)

            sns.heatmap(
                lang_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=lang_emotion_names,
                yticklabels=lang_emotion_names,
                ax=axes[row, col],
                cbar=False,
            )
            axes[row, col].set_title(f"{lang.upper()} (n={len(lang_data)})")
        else:
            axes[row, col].text(
                0.5,
                0.5,
                f"No {lang.upper()} data",
                ha="center",
                va="center",
                transform=axes[row, col].transAxes,
            )
            axes[row, col].set_title(f"{lang.upper()} (n=0)")

        axes[row, col].set_xlabel("Predicted")
        axes[row, col].set_ylabel("True")

    plt.tight_layout()
    summary_save_path = os.path.join(args.save_dir, "confusion_matrix_summary.png")
    plt.savefig(summary_save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved summary plot: {summary_save_path}")
    plt.show()

    print(f"\nAll plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
