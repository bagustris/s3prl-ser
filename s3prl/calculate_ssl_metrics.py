#!/usr/bin/env python3
"""
Calculate metrics for SSL models across 5-fold CV with 128 nodes
"""

import os
import subprocess
import sys
from collections import defaultdict


def run_calc_metric(result_dir):
    """Run calc_metric.py for a given result directory"""
    try:
        # Add trailing slash for glob patterns to work
        result_path = result_dir if result_dir.endswith("/") else result_dir + "/"
        cmd = [sys.executable, "calc_metric.py", result_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Error running calc_metric.py for {result_dir}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception running calc_metric.py for {result_dir}: {e}")
        return None


def parse_metric_output(output):
    """Parse the output from calc_metric.py to extract WA and UA"""
    lines = output.strip().split("\n")
    wa, ua = None, None

    for line in lines:
        if "Weighted accuracy:" in line:
            try:
                wa = float(line.split(":")[-1].strip()) * 100  # Convert to percentage
            except (ValueError, IndexError):
                pass
        elif "Unweighted accuracy:" in line:
            try:
                ua = float(line.split(":")[-1].strip()) * 100  # Convert to percentage
            except (ValueError, IndexError):
                pass

    return wa, ua


def main():
    # SSL models and their corresponding directory patterns
    ssl_models = {
        "UniSpeech-SAT Large": "unispeech_sat_large",
        "WavLM Large": "wavlm_large",
        "XLS-R 300M": "xls_r_300m",
        "XLS-R 1B": "xls_r_1b",
        "XLS-R 2B": "xls_r_2b",
    }

    # Storage for results
    results = defaultdict(list)

    print("Calculating metrics for SSL models with 128 nodes across 5-fold CV...")
    print("=" * 80)

    for model_name, model_pattern in ssl_models.items():
        print(f"\nProcessing {model_name}:")
        fold_results = []

        for fold in range(1, 6):  # folds 1-5
            result_dir = (
                f"result/downstream/emofilm-cv-fold_{fold}.csv-{model_pattern}-128"
            )

            if os.path.exists(result_dir):
                print(f"  Fold {fold}: ", end="")
                output = run_calc_metric(result_dir)

                if output:
                    wa, ua = parse_metric_output(output)
                    if wa is not None and ua is not None:
                        fold_results.append((wa, ua))
                        print(f"WA={wa:.2f}%, UA={ua:.2f}%")
                        results[model_name].append((wa, ua))
                    else:
                        print("Failed to parse metrics")
                else:
                    print("Failed to calculate metrics")
            else:
                print(f"  Fold {fold}: Directory not found - {result_dir}")

        # Calculate average for this model
        if fold_results:
            avg_wa = sum(r[0] for r in fold_results) / len(fold_results)
            avg_ua = sum(r[1] for r in fold_results) / len(fold_results)
            print(f"  Average: WA={avg_wa:.2f}%, UA={avg_ua:.2f}%")
        else:
            print(f"  No valid results found for {model_name}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print("=" * 80)
    print("SSL Model".ljust(25) + "WA".rjust(10) + "UA".rjust(10))
    print("-" * 45)

    for model_name in ssl_models.keys():
        if results[model_name]:
            avg_wa = sum(r[0] for r in results[model_name]) / len(results[model_name])
            avg_ua = sum(r[1] for r in results[model_name]) / len(results[model_name])
            print(
                f"{model_name}".ljust(25)
                + f"{avg_wa:.2f}".rjust(10)
                + f"{avg_ua:.2f}".rjust(10)
            )
        else:
            print(f"{model_name}".ljust(25) + "N/A".rjust(10) + "N/A".rjust(10))

    print("\nNote: All values are percentages")


if __name__ == "__main__":
    main()
