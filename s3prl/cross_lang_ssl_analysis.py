#!/usr/bin/env python3
"""
Cross-Language SSL Performance Analyzer

This script analyzes which SSL models consistently outperform across all languages
and identifies language-specific advantages in emotion recognition.

Usage:
    python cross_lang_ssl_analysis.py --baseline <baseline_dir> --models <model_dirs> --names <model_names>

The script generates:
1. Language-specific performance matrix
2. Consistency ranking across languages
3. Language advantage analysis
4. Overall vs per-language performance comparison
"""

import argparse
from argparse import RawTextHelpFormatter
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score

# Emotion and language mappings
EMOTION_LABELS = {0: "Anger", 1: "Contempt", 2: "Happy", 3: "Neutral", 4: "Sad"}
LANGUAGES = ["en", "it", "es"]
LANGUAGE_NAMES = {"en": "English", "it": "Italian", "es": "Spanish"}


def calculate_language_performance_averaged(
    result_dirs, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate average performance across multiple folds for each language
    """
    all_fold_results = []

    for result_dir in result_dirs:
        fold_result = calculate_language_performance(result_dir, label_csv)
        if fold_result:  # Only add if we got valid results
            all_fold_results.append(fold_result)

    if not all_fold_results:
        return {}

    # Average across folds
    averaged_performance = {}
    for lang in LANGUAGES:
        lang_values = [fold[lang] for fold in all_fold_results if lang in fold]
        if lang_values:
            averaged_performance[lang] = sum(lang_values) / len(lang_values)
        else:
            averaged_performance[lang] = 0.0

    return averaged_performance


def calculate_language_performance(
    result_dir, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """
    Calculate overall accuracy for each language
    """
    # Search test prediction and truth files
    pred_files = glob.glob(os.path.join(result_dir, "test*predict.txt"))
    true_files = glob.glob(os.path.join(result_dir, "test*truth.txt"))

    if not pred_files or not true_files:
        return {}

    pred_file = pred_files[0]
    true_file = true_files[0]

    # Read prediction and truth files
    pred = pd.read_csv(pred_file, header=None)
    true = pd.read_csv(true_file, header=None)

    # Ensure we have Series
    if pred.shape[1] > 1:
        pred = pred.iloc[:, -1]
    else:
        pred = pred.iloc[:, 0]

    if true.shape[1] > 1:
        true = true.iloc[:, -1]
    else:
        true = true.iloc[:, 0]

    # Load the label CSV file
    try:
        df_labels = pd.read_csv(label_csv)
    except FileNotFoundError:
        return {}

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

    # Calculate performance for each language
    lang_performance = {}

    for lang in LANGUAGES:
        lang_data = test_df[test_df["language"] == lang]

        if len(lang_data) == 0:
            lang_performance[lang] = 0.0
            continue

        # Calculate unweighted accuracy (UA) for this language
        lang_pred = lang_data["predicted"]
        lang_true = lang_data["true_label"]

        try:
            ua = balanced_accuracy_score(lang_true, lang_pred)
            lang_performance[lang] = ua
        except Exception:
            lang_performance[lang] = 0.0

    return lang_performance


def create_cross_language_analysis(
    baseline_dir,
    comparison_dirs_list,
    comparison_names,
    save_dir="./",
    label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv",
):
    """
    Create comprehensive cross-language SSL analysis using averaged results from multiple folds

    Args:
        baseline_dir: Single directory for baseline model
        comparison_dirs_list: List of space-separated directory strings, where each string contains directories for one model across folds
        comparison_names: List of model names
    """

    # Calculate baseline performance (assuming single fold for baseline)
    baseline_perf = calculate_language_performance(baseline_dir, label_csv)
    print(f"Baseline (wav2vec2) performance: {baseline_perf}")

    # Calculate comparison model performance (averaged across folds)
    all_performance = {}
    valid_names = []

    for comp_dirs_string, comp_name in zip(comparison_dirs_list, comparison_names):
        try:
            # Split space-separated directories into list
            comp_dirs = comp_dirs_string.split()
            # Average across multiple folds
            comp_perf = calculate_language_performance_averaged(comp_dirs, label_csv)
            if comp_perf:  # Only add if we got valid results
                all_performance[comp_name] = comp_perf
                valid_names.append(comp_name)
                print(f"{comp_name} performance (5-fold average): {comp_perf}")
        except Exception as e:
            print(f"Warning: Could not process {comp_name}: {e}")

    if not all_performance:
        print("Error: No valid comparison models found")
        return

    # Create DataFrame for analysis
    performance_df = pd.DataFrame(all_performance).T
    baseline_df = pd.DataFrame([baseline_perf], index=["wav2vec2"])

    # Calculate improvements over baseline
    improvement_df = performance_df.copy()
    for lang in LANGUAGES:
        if lang in baseline_perf:
            improvement_df[lang] = improvement_df[lang] - baseline_perf[lang]

    # Create individual plots and save each as separate PDF

    # 1. Absolute Performance Heatmap
    plt.figure(figsize=(10, 6))
    combined_df = pd.concat([baseline_df, performance_df])
    sns.heatmap(
        combined_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "Unweighted Accuracy"},
    )
    plt.title("Absolute Performance by Language", fontsize=16, fontweight="bold")
    plt.ylabel("SSL Models", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_path_1 = os.path.join(save_dir, "1_absolute_performance_by_language.pdf")
    plt.savefig(save_path_1, dpi=300, bbox_inches="tight")
    print(f"Saved absolute performance heatmap to: {save_path_1}")
    plt.show()

    # 2. Improvement over wav2vec2 Heatmap
    plt.figure(figsize=(10, 6))
    vmax = max(abs(improvement_df.values.min()), abs(improvement_df.values.max()))
    sns.heatmap(
        improvement_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "Improvement over wav2vec2 (UA)"},
    )
    # Title removed as requested
    plt.ylabel("SSL Models", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_path_2 = os.path.join(save_dir, "2_improvement_over_wav2vec2.pdf")
    plt.savefig(save_path_2, dpi=300, bbox_inches="tight")
    print(f"Saved improvement heatmap to: {save_path_2}")
    plt.show()

    # 3. Consistency Analysis Scatter Plot
    plt.figure(figsize=(10, 6))
    consistency_scores = []
    model_names = []

    for model in valid_names:
        improvements = [
            improvement_df.loc[model, lang]
            for lang in LANGUAGES
            if lang in improvement_df.columns
        ]
        # Consistency = negative of standard deviation (higher = more consistent)
        consistency = -np.std(improvements)
        avg_improvement = np.mean(improvements)
        consistency_scores.append([avg_improvement, consistency])
        model_names.append(model)

    consistency_df = pd.DataFrame(
        consistency_scores,
        index=model_names,
        columns=["Avg Improvement", "Consistency"],
    )

    # Scatter plot: Average improvement vs consistency
    for i, model in enumerate(model_names):
        x_val = float(consistency_df.iloc[i, 0])
        y_val = float(consistency_df.iloc[i, 1])
        plt.scatter(x_val, y_val, s=100)
        plt.annotate(
            model, (x_val, y_val), xytext=(5, 5), textcoords="offset points", fontsize=14
        )

    plt.xlabel("Average Improvement over wav2vec2", fontsize=16)
    plt.ylabel("Consistency (negative std dev)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Performance vs Consistency Trade-off", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path_3 = os.path.join(save_dir, "3_performance_vs_consistency.pdf")
    plt.savefig(save_path_3, dpi=300, bbox_inches="tight")
    print(f"Saved consistency analysis to: {save_path_3}")
    plt.show()

    # 4. Best Model per Language Bar Chart
    plt.figure(figsize=(8, 6))
    best_models = []
    best_improvements = []

    for lang in LANGUAGES:
        if lang in improvement_df.columns:
            best_model = improvement_df[lang].idxmax()
            best_improvement = improvement_df[lang].max()
            best_models.append(best_model)
            best_improvements.append(best_improvement)
        else:
            best_models.append("N/A")
            best_improvements.append(0)

    bars = plt.bar([LANGUAGE_NAMES[lang] for lang in LANGUAGES], best_improvements)
    plt.ylabel("Best Improvement over wav2vec2", fontsize=16)
    plt.title("Best SSL Model per Language", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45)

    # Add model names on bars
    for bar, model in zip(bars, best_models):
        if model != "N/A":
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                model,
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=12,
            )

    plt.tight_layout()
    save_path_4 = os.path.join(save_dir, "4_best_model_per_language.pdf")
    plt.savefig(save_path_4, dpi=300, bbox_inches="tight")
    print(f"Saved best models per language to: {save_path_4}")
    plt.show()

    # 5. Combined Overview (4-panel figure for comparison)
    plt.figure(figsize=(12, 8))

    # Subplot 1: Absolute Performance
    plt.subplot(2, 2, 1)
    sns.heatmap(
        combined_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "UA"},
    )
    plt.title("Absolute Performance")
    plt.ylabel("SSL Models")

    # Subplot 2: Improvement over wav2vec2
    plt.subplot(2, 2, 2)
    sns.heatmap(
        improvement_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "Improvement (UA)"},
    )
    plt.title("Improvement over wav2vec2")
    plt.ylabel("SSL Models")

    # Subplot 3: Consistency Analysis
    plt.subplot(2, 2, 3)
    for i, model in enumerate(model_names):
        x_val = float(consistency_df.iloc[i, 0])
        y_val = float(consistency_df.iloc[i, 1])
        plt.scatter(x_val, y_val, s=100)
        plt.annotate(
            model, (x_val, y_val), xytext=(5, 5), textcoords="offset points", fontsize=14
        )
    plt.xlabel("Avg Improvement", fontsize=14)
    plt.ylabel("Consistency", fontsize=14)
    plt.title("Performance vs Consistency")
    plt.grid(True, alpha=0.3)

    # Subplot 4: Best Model per Language
    plt.subplot(2, 2, 4)
    bars = plt.bar([LANGUAGE_NAMES[lang] for lang in LANGUAGES], best_improvements)
    plt.ylabel("Best Improvement", fontsize=14)
    plt.title("Best SSL Model per Language", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45)

    # Add model names on bars
    for bar, model in zip(bars, best_models):
        if model != "N/A":
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                model,
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=12,
            )

    plt.tight_layout()
    save_path_combined = os.path.join(
        save_dir, "5_cross_language_analysis_combined.pdf"
    )
    plt.savefig(save_path_combined, dpi=300, bbox_inches="tight")
    print(f"Saved combined analysis to: {save_path_combined}")
    plt.show()

    # 6. Language Advantage Analysis
    plt.figure(figsize=(8, 6))
    lang_advantage = improvement_df.copy()
    for model in valid_names:
        model_improvements = improvement_df.loc[model]
        overall_avg = model_improvements.mean()
        for lang in LANGUAGES:
            if lang in lang_advantage.columns:
                lang_advantage.loc[model, lang] = model_improvements[lang] - overall_avg

    sns.heatmap(
        lang_advantage,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "Language Advantage"},
    )
    plt.title(
        "Language-Specific Advantages\n(Above/Below Average Performance)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("SSL Models", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_path_6 = os.path.join(save_dir, "6_language_advantages.pdf")
    plt.savefig(save_path_6, dpi=300, bbox_inches="tight")
    print(f"Saved language advantages to: {save_path_6}")
    plt.show()

    # 7. Ranking Consistency
    plt.figure(figsize=(8, 6))
    rankings = pd.DataFrame(index=valid_names, columns=LANGUAGES)
    for lang in LANGUAGES:
        if lang in improvement_df.columns:
            lang_sorted = improvement_df[lang].sort_values(ascending=False)
            for rank, model in enumerate(lang_sorted.index):
                rankings.loc[model, lang] = rank + 1

    sns.heatmap(
        rankings.astype(float),
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        cbar_kws={"label": "Rank (1=Best)"},
    )
    plt.title(
        "SSL Model Rankings by Language\n(1=Best, Lower is Better)",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("SSL Models", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    save_path_7 = os.path.join(save_dir, "7_model_rankings.pdf")
    plt.savefig(save_path_7, dpi=300, bbox_inches="tight")
    print(f"Saved model rankings to: {save_path_7}")
    plt.show()

    # 8. Combined Language Analysis (2-panel figure)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Language-specific advantage
    sns.heatmap(
        lang_advantage,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        ax=ax1,
        cbar_kws={"label": "Language Advantage"},
    )
    ax1.set_title("Language-Specific Advantages\n(Above/Below Average Performance)")
    ax1.set_ylabel("SSL Models")

    # Plot 2: Ranking consistency
    sns.heatmap(
        rankings.astype(float),
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        ax=ax2,
        cbar_kws={"label": "Rank (1=Best)"},
    )
    ax2.set_title("SSL Model Rankings by Language\n(1=Best, Lower is Better)")
    ax2.set_ylabel("SSL Models")

    plt.tight_layout()
    save_path_8 = os.path.join(save_dir, "8_language_analysis_combined.pdf")
    plt.savefig(save_path_8, dpi=300, bbox_inches="tight")
    print(f"Saved combined language analysis to: {save_path_8}")
    plt.show()

    # 3. Summary Analysis
    print("\n=== CROSS-LANGUAGE ANALYSIS SUMMARY ===")

    # Find most consistent models
    consistency_ranking = consistency_df.sort_values("Consistency", ascending=False)
    print("\nMost Consistent Models (across languages):")
    for i, (model, row) in enumerate(consistency_ranking.head(3).iterrows()):
        print(
            f"  {i+1}. {model}: Avg improvement {row['Avg Improvement']:.3f}, Consistency {row['Consistency']:.3f}"
        )

    # Find best overall improvers
    avg_improvement_ranking = consistency_df.sort_values(
        "Avg Improvement", ascending=False
    )
    print("\nBest Overall Performers:")
    for i, (model, row) in enumerate(avg_improvement_ranking.head(3).iterrows()):
        print(f"  {i+1}. {model}: Avg improvement {row['Avg Improvement']:.3f}")

    # Language-specific best models
    print("\nBest Model per Language:")
    for lang, model, improvement in zip(LANGUAGES, best_models, best_improvements):
        if model != "N/A":
            print(f"  {LANGUAGE_NAMES[lang]}: {model} (+{improvement:.3f})")

    # Check if any model is consistently good across all languages
    print("\nModels in Top-2 for ALL languages:")
    top2_all_langs = []
    for model in valid_names:
        in_top2_count = 0
        for lang in LANGUAGES:
            if lang in rankings.columns:
                rank_val = rankings.loc[model, lang]
                if pd.notna(rank_val) and float(rank_val) <= 2:
                    in_top2_count += 1
        if in_top2_count == len(LANGUAGES):
            top2_all_langs.append(model)

    if top2_all_langs:
        print(f"  {', '.join(top2_all_langs)}")
    else:
        print("  No model consistently in top-2 for all languages")

    # Save detailed results
    results_summary = {
        "absolute_performance": combined_df,
        "improvements": improvement_df,
        "consistency_analysis": consistency_df,
        "language_advantages": lang_advantage,
        "rankings": rankings,
    }
    # Save to CSV files
    for name, df in results_summary.items():
        csv_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(csv_path)
        print(f"Saved {name} data to: {csv_path}")

    print("\n=== Generated Individual PDF Files ===")
    print(f"1. {os.path.join(save_dir, '1_absolute_performance_by_language.pdf')}")
    print(f"2. {os.path.join(save_dir, '2_improvement_over_wav2vec2.pdf')}")
    print(f"3. {os.path.join(save_dir, '3_performance_vs_consistency.pdf')}")
    print(f"4. {os.path.join(save_dir, '4_best_model_per_language.pdf')}")
    print(f"5. {os.path.join(save_dir, '5_cross_language_analysis_combined.pdf')}")
    print(f"6. {os.path.join(save_dir, '6_language_advantages.pdf')}")
    print(f"7. {os.path.join(save_dir, '7_model_rankings.pdf')}")
    print(f"8. {os.path.join(save_dir, '8_language_analysis_combined.pdf')}")

    print("\n=== CSV Data Files ===")
    for name in results_summary.keys():
        print(f"- {name}.csv")

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SSL model performance consistency across languages.\n"
        "Example: python cross_lang_ssl_analysis.py --baseline result/downstream/emofilm-wav2vec2/ \\\n"
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
        "--names", nargs="+", required=True, help="Names of the comparison models"
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

    args = parser.parse_args()

    if len(args.models) != len(args.names):
        print("Error: Number of model directories must match number of model names")
        return

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("Analyzing cross-language SSL performance...")
    print(f"Baseline: wav2vec2 ({args.baseline})")
    print(f"Comparison models: {dict(zip(args.names, args.models))}")
    print(f"Languages: {[LANGUAGE_NAMES[lang] for lang in LANGUAGES]}")

    # Run analysis
    create_cross_language_analysis(
        args.baseline, args.models, args.names, args.save_dir, args.label_csv
    )


if __name__ == "__main__":
    main()
