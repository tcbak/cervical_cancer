import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Split_Data"

OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dataset_distribution.png"


# ==========================================================
# SETTINGS
# ==========================================================

classes = ["CIN1", "CIN2", "CIN3", "Normal"]
splits = ["train", "val", "test"]

# ==========================================================
# COUNT IMAGES
# ==========================================================

counts = {}

for split in splits:
    counts[split] = []

    for class_name in classes:

        class_dir = DATA_DIR / split / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Folder not found:\n{class_dir}"
            )

        image_count = 0

        for file in class_dir.iterdir():
            if file.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            ]:
                image_count += 1

        counts[split].append(image_count)


# ==========================================================
# PRINT DATASET DISTRIBUTION
# ==========================================================

print("=" * 60)
print("CERVICAL CANCER DATASET DISTRIBUTION")
print("=" * 60)

print("\n{:<12} {:>10} {:>10} {:>10}".format(
    "Class", "Train", "Validation", "Test"
))

print("-" * 60)

for i, class_name in enumerate(classes):

    print("{:<12} {:>10} {:>10} {:>10}".format(
        class_name,
        counts["train"][i],
        counts["val"][i],
        counts["test"][i]
    ))

print("-" * 60)

print("{:<12} {:>10} {:>10} {:>10}".format(
    "TOTAL",
    sum(counts["train"]),
    sum(counts["val"]),
    sum(counts["test"])
))

print("\nTotal images:",
      sum(counts["train"]) +
      sum(counts["val"]) +
      sum(counts["test"]))


# ==========================================================
# CREATE GROUPED BAR CHART
# ==========================================================

x = np.arange(len(classes))

width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars_train = ax.bar(
    x - width,
    counts["train"],
    width,
    label="Train Split"
)

bars_val = ax.bar(
    x,
    counts["val"],
    width,
    label="Validation Split"
)

bars_test = ax.bar(
    x + width,
    counts["test"],
    width,
    label="Test Split"
)


# ==========================================================
# LABELS
# ==========================================================

ax.set_xlabel(
    "Class",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Number of Images",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Cervical Cancer Dataset Distribution per Class and Split",
    fontsize=14,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(classes)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)


# ==========================================================
# ADD VALUES ABOVE BARS
# ==========================================================

def add_labels(bars):

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )


add_labels(bars_train)
add_labels(bars_val)
add_labels(bars_test)


# ==========================================================
# SAVE FIGURE
# ==========================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraph saved successfully:")
print(OUTPUT_FILE)