"""
aggregate_results.py
---------------------------------------------

Aggregates evaluation and benchmark results
from all trained models into one comparison
table.

Expected Folder Structure

benchmark_results/

    efficientnet_b0/
        evaluation/
            metrics.json
        benchmark/
            benchmark.json

    mobilenet_v2/
        evaluation/
            metrics.json
        benchmark/
            benchmark.json

    ...

Output

benchmark_results/

    model_comparison.csv

    model_comparison.xlsx
"""

import os
import json
import argparse

import pandas as pd


# ==========================================================
# Helper Functions
# ==========================================================

def load_json(file_path):
    """
    Loads a JSON file.

    Returns None if file does not exist.
    """

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        return json.load(f)


def save_dataframe(df, output_root):
    """
    Saves comparison table as CSV and Excel.
    """

    csv_path = os.path.join(
        output_root,
        "model_comparison.csv"
    )

    excel_path = os.path.join(
        output_root,
        "model_comparison.xlsx"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    df.to_excel(
        excel_path,
        index=False
    )

    print("\nComparison table saved.")

    print(csv_path)

    print(excel_path)

    # ==========================================================
# Flatten Metrics + Benchmark into One Row
# ==========================================================

def flatten_row(metrics, benchmark):
    """
    Combines evaluation metrics and benchmark metrics
    into a single dictionary (one row).
    """

    row = {}

    # ======================================================
    # Evaluation Metrics
    # ======================================================

    if metrics is not None:

        row["Model"] = metrics.get("model_name")

        row["Accuracy (%)"] = metrics.get("accuracy")

        row["Macro Precision"] = metrics.get("precision")

        row["Macro Recall"] = metrics.get("recall")

        row["Macro F1"] = metrics.get("macro_f1")

        row["Weighted F1"] = metrics.get("weighted_f1")

        row["Macro AUC"] = metrics.get("macro_auc")

        row["Test Loss"] = metrics.get("test_loss")

        row["Evaluation Time (s)"] = metrics.get("evaluation_time_seconds")

        row["Test Samples"] = metrics.get("num_test_images")

        per_class = metrics.get("per_class", {})

        for class_name, values in per_class.items():

            row[f"{class_name}_Sensitivity"] = values.get("sensitivity")

            row[f"{class_name}_Specificity"] = values.get("specificity")

    # ======================================================
    # Benchmark Metrics
    # ======================================================

    if benchmark is not None:

        row["Display Name"] = benchmark.get("display_name")

        row["Category"] = benchmark.get("category")

        row["Input Size"] = benchmark.get("input_size")

        row["Parameters"] = benchmark.get("total_parameters")

        row["Trainable Parameters"] = benchmark.get("trainable_parameters")

        row["Parameters (Million)"] = benchmark.get("total_parameters_million")

        row["Model Size (MB)"] = benchmark.get("model_size_mb")

        row["MACs (Million)"] = benchmark.get("macs_million")

        row["FLOPs (Million)"] = benchmark.get("flops_million")

        row["CPU Latency (ms)"] = benchmark.get("cpu_latency_ms")

        row["CPU Throughput (FPS)"] = benchmark.get("cpu_throughput_fps")

        row["GPU Latency (ms)"] = benchmark.get("gpu_latency_ms")

        row["GPU Throughput (FPS)"] = benchmark.get("gpu_throughput_fps")

        row["GPU Peak Memory (MB)"] = benchmark.get("gpu_peak_memory_mb")

        row["Benchmark Time (s)"] = benchmark.get("benchmark_time_seconds")

    # ======================================================
    # Efficiency Score
    # ======================================================

    if (
        row.get("Macro F1") is not None
        and row.get("Parameters (Million)") not in [None, 0]
    ):

        row["F1 per Million Parameters"] = round(

            row["Macro F1"] /
            row["Parameters (Million)"],

            5

        )

    else:

        row["F1 per Million Parameters"] = None

    return row

# ==========================================================
# Aggregate Results
# ==========================================================

def aggregate(args):

    print("=" * 80)
    print("Aggregating Model Results")
    print("=" * 80)

    rows = []

    if not os.path.exists(args.results_root):

        raise FileNotFoundError(

            f"Results folder not found:\n{args.results_root}"

        )

    # ------------------------------------------------------
    # Loop through every model folder
    # ------------------------------------------------------

    for model_name in sorted(os.listdir(args.results_root)):

        model_folder = os.path.join(

            args.results_root,

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

        metrics = load_json(evaluation_file)

        benchmark = load_json(benchmark_file)

        if metrics is None and benchmark is None:

            print(f"Skipping {model_name} (No results found)")

            continue

        print(f"Loading {model_name}")

        row = flatten_row(

            metrics,

            benchmark

        )

        rows.append(row)

    # ------------------------------------------------------
    # Check if anything was loaded
    # ------------------------------------------------------

    if len(rows) == 0:

        print("\nNo benchmark results found.")

        return

    # ------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame(rows)

    # ------------------------------------------------------
    # Sort by Macro F1
    # ------------------------------------------------------

    if "Macro F1" in df.columns:

        df = df.sort_values(

            by="Macro F1",

            ascending=False

        )

    df.reset_index(

        drop=True,

        inplace=True

    )

    # ------------------------------------------------------
    # Save CSV & Excel
    # ------------------------------------------------------

    save_dataframe(

        df,

        args.results_root

    )

    print("\nComparison Summary")

    print(df)

    print("\nFinished.")

    # ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Aggregate all model evaluation and benchmark results"

    )

    parser.add_argument(

        "--results_root",

        type=str,

        default="benchmark_results",

        help="Root directory containing all model folders."

    )

    args = parser.parse_args()

    aggregate(args)