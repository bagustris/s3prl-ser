#!/usr/bin/env python3
"""
SSL Model Comparison - Improvement Delta Heatmap

This script creates a heatmap showing improvement in Unweighted Accuracy (UA) 
for each emotion class across different SSL models relative to wav2vec2 baseline.

Usage:
    python ssl_comparison.py --baseline <wav2vec2_result_dir> --models <model1_dir> <model2_dir> ... --names <name1> <name2> ...

Example:
    python ssl_comparison.py \
        --baseline result/downstream/emofilm-wav2vec2/ \
        --models result/downstream/emofilm-unispeech/ result/downstream/emofilm-wavlm/ \
        --names "UniSpeech-SAT Large" "WavLM Large"

The script will:
1. Calculate per-class UA for baseline and comparison models
2. Generate improvement delta heatmap
3. Save results as PDF files and CSV data
"""

import argparse
from argparse import RawTextHelpFormatter
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Emotion label mapping based on EmoFilm dataset
EMOTION_LABELS = {0: "Anger", 1: "Contempt", 2: "Happy", 3: "Neutral", 4: "Sad"}

# Default SSL model configurations
DEFAULT_MODELS = [
    "UniSpeech-SAT Large",
    "WavLM Large",
    "XLS-R 300M",
    "XLS-R 1B",
    "XLS-R 2B",
]


def calculate_per_class_ua_averaged(
    result_dirs, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate average Unweighted Accuracy (UA) for each emotion class across multiple folds

    Args:
        result_dirs: List of directories containing prediction and truth files for different folds
        label_csv: Path to the EmoFilm label CSV file

    Returns:
        dict: Average UA scores for each emotion class across folds
    """
    all_fold_results = []

    for result_dir in result_dirs:
        fold_result = calculate_per_class_ua(result_dir, label_csv)
        if fold_result:  # Only add if we got valid results
            all_fold_results.append(fold_result)

    if not all_fold_results:
        return {}

    # Get all emotions that appear in any fold
    all_emotions = set()
    for fold in all_fold_results:
        all_emotions.update(fold.keys())

    # Average across folds
    averaged_ua = {}
    for emotion in all_emotions:
        emotion_values = [fold.get(emotion, 0.0) for fold in all_fold_results]
        averaged_ua[emotion] = sum(emotion_values) / len(emotion_values)

    return averaged_ua


def calculate_per_class_ua(
    result_dir, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate Unweighted Accuracy (UA) for each emotion class

    Args:
        result_dir: Directory containing prediction and truth files
        label_csv: Path to the EmoFilm label CSV file

    Returns:
        dict: UA scores for each emotion class
    """
    # Search test prediction and truth files
    pred_files = glob.glob(os.path.join(result_dir, "test*predict.txt"))
    true_files = glob.glob(os.path.join(result_dir, "test*truth.txt"))

    if not pred_files or not true_files:
        raise FileNotFoundError(
            f"Could not find prediction or truth files in {result_dir}"
        )

    pred_file = pred_files[0]
    true_file = true_files[0]

    # Read prediction and truth files
    pred = pd.read_csv(pred_file, header=None)
    true = pd.read_csv(true_file, header=None)

    # Check to use last column and ensure we have Series
    if pred.shape[1] > 1:
        pred = pred.iloc[:, -1]
    else:
        pred = pred.iloc[:, 0]

    if true.shape[1] > 1:
        true = true.iloc[:, -1]
    else:
        true = true.iloc[:, 0]

    # Get unique emotion labels present in the data
    true_unique = true.unique()
    pred_unique = pred.unique()
    all_emotions = sorted(set(list(true_unique) + list(pred_unique)))

    # Calculate per-class UA (recall for each class)
    per_class_ua = {}

    for emotion in all_emotions:
        # Get indices for this emotion class
        true_mask = true == emotion

        if true_mask.sum() == 0:  # No samples for this emotion
            per_class_ua[emotion] = 0.0
            continue

        # Calculate recall (UA) for this class
        correct_predictions = (pred[true_mask] == emotion).sum()
        total_samples = true_mask.sum()
        ua = correct_predictions / total_samples

        per_class_ua[emotion] = ua

    return per_class_ua


def calculate_language_specific_ua_averaged(
    result_dirs, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate average UA for each language and emotion combination across multiple folds

    Returns:
        dict: Nested dict with language -> emotion -> averaged UA
    """
    all_fold_results = []

    for result_dir in result_dirs:
        fold_result = calculate_language_specific_ua(result_dir, label_csv)
        if fold_result:  # Only add if we got valid results
            all_fold_results.append(fold_result)

    if not all_fold_results:
        return {}

    # Get all languages and emotions
    all_languages = set()
    all_emotions_by_lang = {}

    for fold in all_fold_results:
        all_languages.update(fold.keys())
        for lang, emotions in fold.items():
            if lang not in all_emotions_by_lang:
                all_emotions_by_lang[lang] = set()
            all_emotions_by_lang[lang].update(emotions.keys())

    # Average across folds
    averaged_ua = {}
    for lang in all_languages:
        averaged_ua[lang] = {}
        for emotion in all_emotions_by_lang.get(lang, []):
            emotion_values = []
            for fold in all_fold_results:
                if lang in fold and emotion in fold[lang]:
                    emotion_values.append(fold[lang][emotion])
            if emotion_values:
                averaged_ua[lang][emotion] = sum(emotion_values) / len(emotion_values)
            else:
                averaged_ua[lang][emotion] = 0.0

    return averaged_ua


def calculate_language_specific_ua(
    result_dir, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate UA for each language and emotion combination

    Returns:
        dict: Nested dict with language -> emotion -> UA
    """
    # Search test prediction and truth files
    pred_files = glob.glob(os.path.join(result_dir, "test*predict.txt"))
    true_files = glob.glob(os.path.join(result_dir, "test*truth.txt"))

    if not pred_files or not true_files:
        raise FileNotFoundError(
            f"Could not find prediction or truth files in {result_dir}"
        )

    pred_file = pred_files[0]
    true_file = true_files[0]

    # Read prediction and truth files
    pred = pd.read_csv(pred_file, header=None)
    true = pd.read_csv(true_file, header=None)

    # Check to use last column
    if pred.shape[1] > 1:
        pred = pred.iloc[:, -1]
    if true.shape[1] > 1:
        true = true.iloc[:, -1]

    # Load the label CSV file
    try:
        df_labels = pd.read_csv(label_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {label_csv}")

    # Filter for test set only (set == 1)
    test_df = df_labels[df_labels["set"] == 1].reset_index(drop=True)

    if len(test_df) != len(pred):
        min_len = min(len(test_df), len(pred))
        test_df = test_df.iloc[:min_len]
        pred = pred.iloc[:min_len]
        true = true.iloc[:min_len]

    # Add predictions and truth to the dataframe
    test_df = test_df.copy()
    test_df["predicted"] = pred.values
    test_df["true_label"] = true.values

    # Calculate language-specific UA
    languages = ["en", "it", "es"]
    lang_ua = {}

    for lang in languages:
        lang_data = test_df[test_df["language"] == lang]

        if len(lang_data) == 0:
            lang_ua[lang] = {}
            continue

        # Calculate per-class UA for this language
        lang_pred = lang_data["predicted"]
        lang_true = lang_data["true_label"]

        lang_true_unique = lang_true.unique()
        lang_pred_unique = lang_pred.unique()
        all_emotions = sorted(set(list(lang_true_unique) + list(lang_pred_unique)))
        lang_ua[lang] = {}

        for emotion in all_emotions:
            true_mask = lang_true == emotion

            if true_mask.sum() == 0:
                lang_ua[lang][emotion] = 0.0
                continue

            correct_predictions = (lang_pred[true_mask] == emotion).sum()
            total_samples = true_mask.sum()
            ua = correct_predictions / total_samples

            lang_ua[lang][emotion] = ua

    return lang_ua


def create_improvement_heatmap(
    baseline_dir,
    comparison_dirs_list,
    comparison_names,
    save_dir="./",
    label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv",
):
    """
    Create heatmap showing UA improvement over baseline for each emotion

    Args:
        baseline_dir: Single directory for baseline model
        comparison_dirs_list: List of space-separated directory strings for each model across folds
        comparison_names: List of model names
    """
    # Get baseline UA scores (assuming single directory for baseline)
    try:
        baseline_ua = calculate_per_class_ua(baseline_dir, label_csv)
        print(f"Baseline (wav2vec2) UA scores: {baseline_ua}")
    except Exception as e:
        print(f"Error processing baseline: {e}")
        return

    # Get all emotion labels from baseline
    all_emotions = sorted(baseline_ua.keys())
    emotion_names = [EMOTION_LABELS.get(emo, f"Emotion_{emo}") for emo in all_emotions]

    # Calculate improvements for each model (averaged across folds)
    improvements = []
    valid_names = []

    for comp_dirs_string, comp_name in zip(comparison_dirs_list, comparison_names):
        try:
            # Split space-separated directories into list
            comp_dirs = comp_dirs_string.split()
            comp_ua = calculate_per_class_ua_averaged(comp_dirs, label_csv)
            print(f"{comp_name} UA scores (5-fold average): {comp_ua}")

            # Calculate improvement for each emotion
            improvement = []
            for emotion in all_emotions:
                baseline_score = baseline_ua.get(emotion, 0.0)
                comp_score = comp_ua.get(emotion, 0.0)
                improvement.append(comp_score - baseline_score)

            improvements.append(improvement)
            valid_names.append(comp_name)

        except Exception as e:
            print(f"Warning: Could not process {comp_name}: {e}")

    if not improvements:
        print("Error: No valid comparison models found")
        return

    # Create DataFrame
    df = pd.DataFrame(improvements, index=valid_names, columns=emotion_names)

    # Save raw data
    df.to_csv(os.path.join(save_dir, "ssl_ua_improvements.csv"))
    print(
        f"Saved raw improvement data to: {os.path.join(save_dir, 'ssl_ua_improvements.csv')}"
    )

    # Create heatmap with diverging colormap
    plt.figure(figsize=(12, 8))

    # Use a diverging colormap centered at 0
    vmax = max(abs(df.values.min()), abs(df.values.max()))

    sns.heatmap(
        df,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "UA Improvement over wav2vec2"},
    )

    plt.title(
        "SSL Model Performance: UA Improvement over wav2vec2 Baseline\n(Per Emotion Class)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    # plt.ylabel("SSL Models", fontsize=14)
    plt.xlabel("Emotion Classes", fontsize=14)

    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, "ssl_ua_improvement_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved improvement heatmap to: {save_path}")

    plt.show()

    return df


def create_language_comparison_heatmap(
    baseline_dir,
    comparison_dirs_list,
    comparison_names,
    save_dir="./",
    label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv",
):
    """
    Create heatmaps showing UA improvement by language using 5-fold averaging

    Args:
        baseline_dir: Single directory for baseline model
        comparison_dirs_list: List of space-separated directory strings for each model across folds
        comparison_names: List of model names
    """
    languages = ["en", "it", "es"]

    # Get baseline language-specific UA
    try:
        baseline_lang_ua = calculate_language_specific_ua(baseline_dir, label_csv)
    except Exception as e:
        print(f"Error processing baseline for language analysis: {e}")
        return

    # Create subplot for each language
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for lang_idx, lang in enumerate(languages):
        lang_improvements = []
        valid_names = []

        # Get all emotions for this language from baseline
        lang_emotions = sorted(baseline_lang_ua.get(lang, {}).keys())
        emotion_names = [
            EMOTION_LABELS.get(emo, f"Emotion_{emo}") for emo in lang_emotions
        ]

        # Calculate improvements for each model (averaged across folds)
        for comp_dirs_string, comp_name in zip(comparison_dirs_list, comparison_names):
            try:
                # Split space-separated directories into list
                comp_dirs = comp_dirs_string.split()
                comp_lang_ua = calculate_language_specific_ua_averaged(
                    comp_dirs, label_csv
                )

                improvement = []
                for emotion in lang_emotions:
                    baseline_score = baseline_lang_ua.get(lang, {}).get(emotion, 0.0)
                    comp_score = comp_lang_ua.get(lang, {}).get(emotion, 0.0)
                    improvement.append(comp_score - baseline_score)

                lang_improvements.append(improvement)
                valid_names.append(comp_name)

            except Exception as e:
                print(f"Warning: Could not process {comp_name} for {lang}: {e}")

        if not lang_improvements:
            axes[lang_idx].text(
                0.5,
                0.5,
                f"No data for {lang.upper()}",
                ha="center",
                va="center",
                transform=axes[lang_idx].transAxes,
            )
            axes[lang_idx].set_title(f"{lang.upper()}")
            continue

        # Create DataFrame for this language
        df_lang = pd.DataFrame(
            lang_improvements, index=valid_names, columns=emotion_names
        )

        # Create heatmap
        vmax = max(abs(df_lang.values.min()), abs(df_lang.values.max()))

        sns.heatmap(
            df_lang,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
            vmin=-vmax,
            vmax=vmax,
            ax=axes[lang_idx],
            cbar=lang_idx == 2,
        )  # Only show colorbar on last subplot

        axes[lang_idx].set_title(
            f"{lang.upper()} (n={len(baseline_lang_ua.get(lang, {}))} classes)"
        )
        if lang_idx == 0:
            axes[lang_idx].set_ylabel("SSL Models", fontsize=16)
            axes[lang_idx].tick_params(axis='y', labelsize=14)
        else:
            axes[lang_idx].set_ylabel("")
            axes[lang_idx].set_yticklabels([])
        axes[lang_idx].set_xlabel("Emotion Classes", fontsize=16)
        axes[lang_idx].tick_params(axis='x', labelsize=14)

    # plt.suptitle(
    #     "SSL Model UA Improvement by Emotion Categoris (vs wav2vec2)",
    #     fontsize=16,
    #     fontweight="bold",
    # )
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, "ssl_ua_improvement_by_emotion.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved language-specific improvement heatmap to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SSL model comparison heatmaps using Unweighted Accuracy (UA).\n"
        "Shows improvement over wav2vec2 baseline for different emotions.\n"
        "Example: python ssl_comparison.py --baseline result/downstream/emofilm-wav2vec2/ \\\n"
        "         --models result/downstream/emofilm-unispeech/ result/downstream/emofilm-wavlm/ \\\n"
        "         --names 'UniSpeech-SAT Large' 'WavLM Large'\n"
        "Author: bagus.tris@naist.ac.jp",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Directory containing wav2vec2 baseline results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Directories containing comparison model results",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Names of the comparison models",
    )
    parser.add_argument(
        "--label-csv",
        type=str,
        default="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv",
        help="Path to the EmoFilm label CSV file",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./", help="Directory to save plots and data"
    )
    parser.add_argument(
        "--include-language",
        action="store_true",
        help="Also generate language-specific comparison heatmaps",
    )

    args = parser.parse_args()

    # Ensure we have the same number of models and names
    if len(args.models) != len(args.names):
        # If names weren't provided by user, use defaults
        if args.names == DEFAULT_MODELS:
            # Use default names up to the number of models provided
            args.names = DEFAULT_MODELS[: len(args.models)]
        else:
            print("Error: Number of model directories must match number of model names")
            print(f"Found {len(args.models)} models but {len(args.names)} names")
            print(f"Models: {args.models}")
            print(f"Names: {args.names}")
            return

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Baseline: wav2vec2 ({args.baseline})")
    print(f"Comparison models: {dict(zip(args.names, args.models))}")
    print(f"Label CSV: {args.label_csv}")
    print(f"Save directory: {args.save_dir}")

    # Generate overall improvement heatmap
    print("\n=== Generating Overall UA Improvement Heatmap ===")
    df = create_improvement_heatmap(
        args.baseline, args.models, args.names, args.save_dir, args.label_csv
    )

    if df is not None:
        print("\nImprovement Summary:")
        print(df)

        # Print best performing model per emotion
        print("\nBest performing SSL model per emotion:")
        for emotion in df.columns:
            best_model = df[emotion].idxmax()
            best_improvement = df[emotion].max()
            print(f"  {emotion}: {best_model} (+{best_improvement:.3f})")

    # Generate language-specific heatmaps if requested
    if args.include_language:
        print("\n=== Generating Language-Specific UA Improvement Heatmaps ===")
        create_language_comparison_heatmap(
            args.baseline, args.models, args.names, args.save_dir, args.label_csv
        )


if __name__ == "__main__":
    main()
