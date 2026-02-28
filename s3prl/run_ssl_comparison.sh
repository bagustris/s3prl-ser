#!/bin/bash
# Example script to run SSL comparison analysis
# 
# This script demonstrates how to use ssl_comparison.py to compare
# different SSL models against wav2vec2 baseline for EmoFilm emotion recognition
#
# Prerequisites:
# 1. Run experiments with different SSL models and save results in result/downstream/
# 2. Ensure each result directory contains test*predict.txt and test*truth.txt files
# 3. Have the EmoFilm label CSV file available

# Set paths - using all 5 folds for averaging
BASELINE_DIR="result/downstream/emofilm-wav2vec2_large_ll60k-128"
SAVE_DIR="plots/ssl_comparison_5folds"

# SSL model result directories - all 5 folds
UNISPEECH_DIRS=(
    "result/downstream/emofilm-cv-fold_1.csv-unispeech_sat_large-128"
    "result/downstream/emofilm-cv-fold_2.csv-unispeech_sat_large-128"
    "result/downstream/emofilm-cv-fold_3.csv-unispeech_sat_large-128"
    "result/downstream/emofilm-cv-fold_4.csv-unispeech_sat_large-128"
    "result/downstream/emofilm-cv-fold_5.csv-unispeech_sat_large-128"
)

WAVLM_DIRS=(
    "result/downstream/emofilm-cv-fold_1.csv-wavlm_large-128"
    "result/downstream/emofilm-cv-fold_2.csv-wavlm_large-128"
    "result/downstream/emofilm-cv-fold_3.csv-wavlm_large-128"
    "result/downstream/emofilm-cv-fold_4.csv-wavlm_large-128"
    "result/downstream/emofilm-cv-fold_5.csv-wavlm_large-128"
)

XLSR_300M_DIRS=(
    "result/downstream/emofilm-cv-fold_1.csv-xls_r_300m-128"
    "result/downstream/emofilm-cv-fold_2.csv-xls_r_300m-128"
    "result/downstream/emofilm-cv-fold_3.csv-xls_r_300m-128"
    "result/downstream/emofilm-cv-fold_4.csv-xls_r_300m-128"
    "result/downstream/emofilm-cv-fold_5.csv-xls_r_300m-128"
)

XLSR_1B_DIRS=(
    "result/downstream/emofilm-cv-fold_1.csv-xls_r_1b-128"
    "result/downstream/emofilm-cv-fold_2.csv-xls_r_1b-128"
    "result/downstream/emofilm-cv-fold_3.csv-xls_r_1b-128"
    "result/downstream/emofilm-cv-fold_4.csv-xls_r_1b-128"
    "result/downstream/emofilm-cv-fold_5.csv-xls_r_1b-128"
)

XLSR_2B_DIRS=(
    "result/downstream/emofilm-cv-fold_1.csv-xls_r_2b-128"
    "result/downstream/emofilm-cv-fold_2.csv-xls_r_2b-128"
    "result/downstream/emofilm-cv-fold_3.csv-xls_r_2b-128"
    "result/downstream/emofilm-cv-fold_4.csv-xls_r_2b-128"
    "result/downstream/emofilm-cv-fold_5.csv-xls_r_2b-128"
)

# Create output directory
mkdir -p "$SAVE_DIR"

echo "=== SSL Model Comparison Analysis ==="
echo "Baseline: wav2vec2 ($BASELINE_DIR)"
echo "Output directory: $SAVE_DIR"
echo ""

# Check if baseline exists
if [ ! -d "$BASELINE_DIR" ]; then
    echo "Error: Baseline directory not found: $BASELINE_DIR"
    echo "Please run wav2vec2 experiment first or update the path"
    exit 1
fi

# Function to check if at least one directory in array exists
check_model_dirs() {
    local -n dirs=$1
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            return 0
        fi
    done
    return 1
}

# Collect available model directories (check arrays)
MODELS_ARGS=()
NAMES=()

if check_model_dirs UNISPEECH_DIRS; then
    # Add all directories for this model as a space-separated string
    MODELS_ARGS+=("${UNISPEECH_DIRS[*]}")
    NAMES+=("UniSpeech-SAT Large")
    echo "Found: UniSpeech-SAT Large (${#UNISPEECH_DIRS[@]} folds)"
fi

if check_model_dirs WAVLM_DIRS; then
    MODELS_ARGS+=("${WAVLM_DIRS[*]}")
    NAMES+=("WavLM Large")
    echo "Found: WavLM Large (${#WAVLM_DIRS[@]} folds)"
fi

if check_model_dirs XLSR_300M_DIRS; then
    MODELS_ARGS+=("${XLSR_300M_DIRS[*]}")
    NAMES+=("XLS-R 300M")
    echo "Found: XLS-R 300M (${#XLSR_300M_DIRS[@]} folds)"
fi

if check_model_dirs XLSR_1B_DIRS; then
    MODELS_ARGS+=("${XLSR_1B_DIRS[*]}")
    NAMES+=("XLS-R 1B")
    echo "Found: XLS-R 1B (${#XLSR_1B_DIRS[@]} folds)"
fi

if check_model_dirs XLSR_2B_DIRS; then
    MODELS_ARGS+=("${XLSR_2B_DIRS[*]}")
    NAMES+=("XLS-R 2B")
    echo "Found: XLS-R 2B (${#XLSR_2B_DIRS[@]} folds)"
fi

if [ ${#MODELS_ARGS[@]} -eq 0 ]; then
    echo "Error: No comparison model directories found"
    echo "Please run experiments with SSL models or update the paths"
    exit 1
fi

echo ""
echo "Found ${#MODELS_ARGS[@]} comparison models"
echo ""

# Run the comparison with both basic and language-specific analysis
echo "=== Generating SSL Comparison Analysis ==="
python ssl_comparison.py \
    --baseline "$BASELINE_DIR" \
    --models "${MODELS_ARGS[@]}" \
    --names "${NAMES[@]}" \
    --save-dir "$SAVE_DIR" \
    --include-language

echo ""
echo "=== Analysis Complete ==="
echo "Results saved to: $SAVE_DIR"
echo ""
echo "Generated files:"
echo "- ssl_ua_improvement_heatmap.pdf (main comparison)"
echo "- ssl_ua_improvements.csv (raw data)"
echo "- ssl_ua_improvement_by_emotion.pdf (language breakdown)"
echo ""
echo "Open the PDF files to see the visual comparison results!"
