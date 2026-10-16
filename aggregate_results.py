
import os
import json
import argparse

import pandas as pd


# ==========================================================
# Helper Functions
# ==========================================================

def load_json(file_path):
    """Load a JSON file. Return None if it does not exist."""

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df, output_root):
    """Save comparison results as CSV and Excel."""

    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(
        output_root,
        "model_comparison.csv"
    )

    excel_path = os.path.join(
        output_root,
        "model_comparison.xlsx"
    )

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print("\nComparison files saved:")
    print(csv_path)
    print(excel_path)


# ==========================================================
# Flatten Metrics + Benchmark into One Row
# ==========================================================

def flatten_row(model_name, metrics, benchmark, per_class_metrics=None):
    """Combine model evaluation and benchmark results into one row."""

    row = {
        "Model": model_name,
    }

    metrics = metrics or {}
    benchmark = benchmark or {}
    per_class_metrics = per_class_metrics or {}

    # ------------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------------

    accuracy = metrics.get("accuracy")

    row["Accuracy (%)"] = (
        round(accuracy * 100, 2)
        if isinstance(accuracy, (int, float))
        else None
    )

    row["Macro Precision"] = metrics.get("macro_precision")
    row["Macro Recall"] = metrics.get("macro_recall")
    row["Macro F1"] = metrics.get("macro_f1")
    row["Weighted F1"] = metrics.get("weighted_f1")
    row["Macro AUC"] = metrics.get("macro_auc")
    row["Test Loss"] = metrics.get("test_loss")
    row["Evaluation Time (s)"] = metrics.get("evaluation_time")

    # ------------------------------------------------------
    # Per-Class Sensitivity and Specificity
    # ------------------------------------------------------

    for class_name, class_values in per_class_metrics.items():

        if not isinstance(class_values, dict):
            continue

        row[f"{class_name} Sensitivity"] = (
            class_values.get("sensitivity")
        )

        row[f"{class_name} Specificity"] = (
            class_values.get("specificity")
        )

    # ------------------------------------------------------
    # Benchmark Metrics
    # ------------------------------------------------------

    # Keep the benchmark fields in the CSV.
    # These names should match the keys in benchmark.json.
    row["Total Parameters"] = benchmark.get("total_parameters")
    row["Trainable Parameters"] = benchmark.get("trainable_parameters")
    row["Model Size (MB)"] = benchmark.get("model_size_mb")
    row["CPU Latency (ms)"] = benchmark.get("cpu_latency_ms")
    row["CPU FPS"] = benchmark.get("cpu_fps")
    row["GPU Latency (ms)"] = benchmark.get("gpu_latency_ms")
    row["GPU FPS"] = benchmark.get("gpu_fps")
    row["Peak GPU Memory (MB)"] = benchmark.get("peak_gpu_memory_mb")
    row["FLOPs"] = benchmark.get("flops")
    row["MACs"] = benchmark.get("macs")

    # Add any other benchmark fields without overwriting
    # the explicitly named columns above.
    for key, value in benchmark.items():
        if key not in row:
            row[key] = value

    return row


# ==========================================================
# Aggregate Results
# ==========================================================

def aggregate(args):

    print("=" * 80)
    print("Aggregating Model Results")
    print("=" * 80)

    results_root = os.path.abspath(args.results_root)

    if not os.path.isdir(results_root):
        raise FileNotFoundError(
            f"Results folder not found:\n{results_root}"
        )

    rows = []

    # ------------------------------------------------------
    # Loop through model folders
    # ------------------------------------------------------

    for model_name in sorted(os.listdir(results_root)):

        model_folder = os.path.join(
            results_root,
            model_name
        )

        if not os.path.isdir(model_folder):
            continue

        evaluation_file = os.path.join(
            model_folder,
            "evaluation",
            "metrics",
            "metrics.json"
        )

        benchmark_file = os.path.join(
            model_folder,
            "benchmark",
            "benchmark.json"
        )

        # Load evaluation and benchmark files
        metrics = load_json(evaluation_file)
        benchmark = load_json(benchmark_file)

        # Load per-class sensitivity and specificity
        per_class_file = os.path.join(
            model_folder,
            "evaluation",
            "metrics",
            "per_class_metrics.json"
        )

        per_class_metrics = load_json(per_class_file)

        if metrics is None and benchmark is None:
            print(f"Skipping {model_name} (No results found)")
            continue

        print(f"Loading {model_name}")

        row = flatten_row(
            model_name,
            metrics,
            benchmark,
            per_class_metrics
        )

        rows.append(row)

    # ------------------------------------------------------
    # Check results
    # ------------------------------------------------------

    if not rows:
        print("\nNo evaluation or benchmark results found.")
        return

    # ------------------------------------------------------
    # Create comparison DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame(rows)

    # Sort by Macro F1 when available
    if "Macro F1" in df.columns:
        df = df.sort_values(
            by="Macro F1",
            ascending=False,
            na_position="last"
        )

    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------
    # Save CSV and Excel
    # ------------------------------------------------------

    save_dataframe(
        df,
        results_root
    )

    print("\nComparison Summary")
    print(df.to_string(index=False))

    print("\nFinished successfully.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate model evaluation and benchmark results."
        )
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="benchmark_results",
        help="Root directory containing model result folders."
    )

    args = parser.parse_args()

    aggregate(args)
