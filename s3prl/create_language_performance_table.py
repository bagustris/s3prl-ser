#!/usr/bin/env python3
"""
Create comprehensive SSL performance table by language (EN, IT, ES)
Including wav2vec2 baseline + 5 SSL models with WA and UA metrics
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Model configurations
MODELS = {
    "wav2vec2": {
        "name": "wav2vec2 Large",
        "dirs": ["result/downstream/emofilm-wav2vec2_large_ll60k-128"],
    },
    "unispeech": {
        "name": "UniSpeech-SAT Large",
        "dirs": [
            "result/downstream/emofilm-cv-fold_1.csv-unispeech_sat_large-128",
            "result/downstream/emofilm-cv-fold_2.csv-unispeech_sat_large-128",
            "result/downstream/emofilm-cv-fold_3.csv-unispeech_sat_large-128",
            "result/downstream/emofilm-cv-fold_4.csv-unispeech_sat_large-128",
            "result/downstream/emofilm-cv-fold_5.csv-unispeech_sat_large-128",
        ],
    },
    "wavlm": {
        "name": "WavLM Large",
        "dirs": [
            "result/downstream/emofilm-cv-fold_1.csv-wavlm_large-128",
            "result/downstream/emofilm-cv-fold_2.csv-wavlm_large-128",
            "result/downstream/emofilm-cv-fold_3.csv-wavlm_large-128",
            "result/downstream/emofilm-cv-fold_4.csv-wavlm_large-128",
            "result/downstream/emofilm-cv-fold_5.csv-wavlm_large-128",
        ],
    },
    "xlsr_300m": {
        "name": "XLS-R 300M",
        "dirs": [
            "result/downstream/emofilm-cv-fold_1.csv-xls_r_300m-128",
            "result/downstream/emofilm-cv-fold_2.csv-xls_r_300m-128",
            "result/downstream/emofilm-cv-fold_3.csv-xls_r_300m-128",
            "result/downstream/emofilm-cv-fold_4.csv-xls_r_300m-128",
            "result/downstream/emofilm-cv-fold_5.csv-xls_r_300m-128",
        ],
    },
    "xlsr_1b": {
        "name": "XLS-R 1B",
        "dirs": [
            "result/downstream/emofilm-cv-fold_1.csv-xls_r_1b-128",
            "result/downstream/emofilm-cv-fold_2.csv-xls_r_1b-128",
            "result/downstream/emofilm-cv-fold_3.csv-xls_r_1b-128",
            "result/downstream/emofilm-cv-fold_4.csv-xls_r_1b-128",
            "result/downstream/emofilm-cv-fold_5.csv-xls_r_1b-128",
        ],
    },
    "xlsr_2b": {
        "name": "XLS-R 2B",
        "dirs": [
            "result/downstream/emofilm-cv-fold_1.csv-xls_r_2b-128",
            "result/downstream/emofilm-cv-fold_2.csv-xls_r_2b-128",
            "result/downstream/emofilm-cv-fold_3.csv-xls_r_2b-128",
            "result/downstream/emofilm-cv-fold_4.csv-xls_r_2b-128",
            "result/downstream/emofilm-cv-fold_5.csv-xls_r_2b-128",
        ],
    },
}

LANGUAGES = ["en", "it", "es"]
LANGUAGE_NAMES = {"en": "English", "it": "Italian", "es": "Spanish"}


def load_predictions_and_labels(result_dir):
    """Load predictions and truth labels from result directory"""
    pred_files = glob.glob(os.path.join(result_dir, "test*predict.txt"))
    true_files = glob.glob(os.path.join(result_dir, "test*truth.txt"))

    if not pred_files or not true_files:
        return None, None

    pred_file = pred_files[0]
    true_file = true_files[0]

    pred = pd.read_csv(pred_file, header=None)
    true = pd.read_csv(true_file, header=None)

    # Ensure we get the right columns
    if pred.shape[1] > 1:
        pred = pred.iloc[:, -1]
    else:
        pred = pred.iloc[:, 0]

    if true.shape[1] > 1:
        true = true.iloc[:, -1]
    else:
        true = true.iloc[:, 0]

    return pred.values, true.values


def calculate_language_metrics(
    result_dirs, label_csv="downstream/emofilm/EmoFilm_labels_16k_train_dev_split.csv"
):
    """Calculate WA and UA metrics for each language across folds"""

    # Load label CSV
    try:
        df_labels = pd.read_csv(label_csv)
    except FileNotFoundError:
        print(f"Error: Could not find label file {label_csv}")
        return {}

    # Filter for test set
    test_df = df_labels[df_labels["set"] == 1].reset_index(drop=True)

    language_results = {lang: {"WA": [], "UA": []} for lang in LANGUAGES}

    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            continue

        pred, true = load_predictions_and_labels(result_dir)
        if pred is None or true is None:
            continue

        # Ensure same length
        min_len = min(len(test_df), len(pred), len(true))
        test_df_fold = test_df.iloc[:min_len].copy()
        pred_fold = pred[:min_len]
        true_fold = true[:min_len]

        # Add predictions to test dataframe
        test_df_fold["predicted"] = pred_fold
        test_df_fold["true_label"] = true_fold

        # Calculate metrics per language
        for lang in LANGUAGES:
            lang_data = test_df_fold[test_df_fold["language"] == lang]

            if len(lang_data) == 0:
                continue

            lang_pred = lang_data["predicted"].values
            lang_true = lang_data["true_label"].values

            # Calculate WA and UA
            wa = accuracy_score(lang_true, lang_pred) * 100
            ua = balanced_accuracy_score(lang_true, lang_pred) * 100

            language_results[lang]["WA"].append(wa)
            language_results[lang]["UA"].append(ua)

    # Average across folds
    averaged_results = {}
    for lang in LANGUAGES:
        if language_results[lang]["WA"]:
            averaged_results[lang] = {
                "WA": np.mean(language_results[lang]["WA"]),
                "UA": np.mean(language_results[lang]["UA"]),
            }
        else:
            averaged_results[lang] = {"WA": 0.0, "UA": 0.0}

    return averaged_results


def create_performance_table():
    """Create comprehensive performance table"""

    print("=" * 80)
    print("SSL MODEL PERFORMANCE BY LANGUAGE (WA & UA)")
    print("=" * 80)

    # Store all results
    all_results = {}

    # Process each model
    for model_key, model_config in MODELS.items():
        print(f"\nProcessing {model_config['name']}...")

        lang_metrics = calculate_language_metrics(model_config["dirs"])
        all_results[model_config["name"]] = lang_metrics

        # Print results for this model
        for lang in LANGUAGES:
            if lang in lang_metrics:
                wa = lang_metrics[lang]["WA"]
                ua = lang_metrics[lang]["UA"]
                print(f"  {LANGUAGE_NAMES[lang]}: WA={wa:.2f}%, UA={ua:.2f}%")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE TABLE")
    print("=" * 80)

    # Create formatted table
    header = f"{'Model':<25} {'EN-WA':<8} {'EN-UA':<8} {'IT-WA':<8} {'IT-UA':<8} {'ES-WA':<8} {'ES-UA':<8}"
    print(header)
    print("-" * len(header))

    for model_name, lang_results in all_results.items():
        en_wa = lang_results.get("en", {}).get("WA", 0.0)
        en_ua = lang_results.get("en", {}).get("UA", 0.0)
        it_wa = lang_results.get("it", {}).get("WA", 0.0)
        it_ua = lang_results.get("it", {}).get("UA", 0.0)
        es_wa = lang_results.get("es", {}).get("WA", 0.0)
        es_ua = lang_results.get("es", {}).get("UA", 0.0)

        row = f"{model_name:<25} {en_wa:<8.2f} {en_ua:<8.2f} {it_wa:<8.2f} {it_ua:<8.2f} {es_wa:<8.2f} {es_ua:<8.2f}"
        print(row)

    # Save to CSV
    csv_data = []
    for model_name, lang_results in all_results.items():
        row = {
            "Model": model_name,
            "EN_WA": lang_results.get("en", {}).get("WA", 0.0),
            "EN_UA": lang_results.get("en", {}).get("UA", 0.0),
            "IT_WA": lang_results.get("it", {}).get("WA", 0.0),
            "IT_UA": lang_results.get("it", {}).get("UA", 0.0),
            "ES_WA": lang_results.get("es", {}).get("WA", 0.0),
            "ES_UA": lang_results.get("es", {}).get("UA", 0.0),
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = "ssl_language_performance_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Find best model per language per metric
    for lang in LANGUAGES:
        print(f"\n{LANGUAGE_NAMES[lang]}:")

        # Best WA
        best_wa_model = max(
            all_results.items(), key=lambda x: x[1].get(lang, {}).get("WA", 0.0)
        )
        best_wa_score = best_wa_model[1].get(lang, {}).get("WA", 0.0)
        print(f"  Best WA: {best_wa_model[0]} ({best_wa_score:.2f}%)")

        # Best UA
        best_ua_model = max(
            all_results.items(), key=lambda x: x[1].get(lang, {}).get("UA", 0.0)
        )
        best_ua_score = best_ua_model[1].get(lang, {}).get("UA", 0.0)
        print(f"  Best UA: {best_ua_model[0]} ({best_ua_score:.2f}%)")

    return all_results


if __name__ == "__main__":
    create_performance_table()
