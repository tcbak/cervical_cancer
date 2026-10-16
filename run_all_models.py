
from pathlib import Path
import subprocess
import sys

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "Split_Data_PatientWise"
RESULTS_ROOT = PROJECT_ROOT / "outputs_merged_patientwise_20261013"

TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "evaluate.py"
BENCHMARK_SCRIPT = PROJECT_ROOT / "benchmark.py"
AGGREGATE_SCRIPT = PROJECT_ROOT / "aggregate_results.py"

# Start with one model to verify the complete pipeline.
# After success, add the remaining model names.
MODELS = [

    "efficientnet_b0",

    "mobilenet_v2",

    "shufflenet_v2_x1_0",

    "squeezenet1_1",

    "densenet121",

    "resnet18",

    "resnet34",

    "resnet50",

    "vgg16"

]

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5


def run_command(command):
    """Run a script and stop immediately if it fails."""
    print("\n" + "=" * 80)
    print("Running:", " ".join(str(item) for item in command))
    print("=" * 80)

    subprocess.run(
        [str(item) for item in command],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def validate_paths():
    """Check the required inputs before starting training."""

    required_paths = [
        DATASET_PATH / "Train",
        DATASET_PATH / "Validation",
        DATASET_PATH / "Test",
        TRAIN_SCRIPT,
        EVALUATE_SCRIPT,
        BENCHMARK_SCRIPT,
        AGGREGATE_SCRIPT,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    for split in ["Train", "Validation", "Test"]:
        split_path = DATASET_PATH / split

        class_folders = [
            item for item in split_path.iterdir()
            if item.is_dir()
        ]

        if not class_folders:
            raise RuntimeError(
                f"No class folders found in:\n{split_path}"
            )

    # Prevent accidental overwriting of a previous experiment.
    if RESULTS_ROOT.exists() and any(RESULTS_ROOT.iterdir()):
        raise FileExistsError(
            "The output folder already contains files:\n"
            f"{RESULTS_ROOT}\n"
            "Choose a new RESULTS_ROOT before running."
        )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    validate_paths()

    print("Project:", PROJECT_ROOT)
    print("Dataset:", DATASET_PATH)
    print("Results:", RESULTS_ROOT)
    print("Models:", MODELS)

    for model_name in MODELS:
        print(f"\n\n######## {model_name} ########")

        # 1. Train with the existing training strategy.
        run_command([
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model_name", model_name,
            "--data_dir", str(DATASET_PATH),
            "--output_dir", str(RESULTS_ROOT),
            "--batch_size", str(BATCH_SIZE),
            "--epochs", str(EPOCHS),
            "--lr", str(LEARNING_RATE),
            "--weight_decay", str(WEIGHT_DECAY),
            "--patience", str(PATIENCE),
        ])

        checkpoint = (
            RESULTS_ROOT
            / model_name
            / "checkpoints"
            / "best_model.pth"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "Training did not produce the expected checkpoint:\n"
                f"{checkpoint}"
            )

        # 2. Evaluate the saved best checkpoint on the test set.
        run_command([
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--model_name", model_name,
            "--data_dir", str(DATASET_PATH),
            "--model_path", str(checkpoint),
            "--output_dir",
            str(RESULTS_ROOT / model_name / "evaluation"),
            "--batch_size", str(BATCH_SIZE),
        ])

        # 3. Benchmark the model architecture.
        run_command([
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--model_name", model_name,
            "--output_dir", str(RESULTS_ROOT),
        ])

    # 4. Combine model evaluation and benchmark results.
    run_command([
        sys.executable,
        str(AGGREGATE_SCRIPT),
        "--results_root", str(RESULTS_ROOT),
    ])

    print("\nPipeline completed successfully.")
    print("Results folder:", RESULTS_ROOT)
    print("Comparison CSV:", RESULTS_ROOT / "model_comparison.csv")
    print("Comparison Excel:", RESULTS_ROOT / "model_comparison.xlsx")


if __name__ == "__main__":
    main()
