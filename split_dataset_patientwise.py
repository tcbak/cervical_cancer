
from pathlib import Path
from collections import defaultdict
import csv
import random
import shutil
import sys

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "merged_dataset"
OUTPUT_DIR = PROJECT_ROOT / "Split_Data_PatientWise"

# Desired split proportions
SPLIT_RATIOS = {
    "Train": 0.70,
    "Validation": 0.15,
    "Test": 0.15,
}

# Reproducible results: the same source and seed give the
# same assignment of patients to splits.
RANDOM_SEED = 42

CLASSES = ["Normal", "CIN1", "CIN2", "CIN3"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):
    """Find supported images recursively inside a folder."""
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# DISCOVER PATIENTS
# ============================================================

def discover_patients():
    """
    Read the merged dataset.

    Expected structure:
        merged_dataset/
            Normal/
                patient_id/
                    image.jpg
            CIN1/
                patient_id/
                    image.jpg
            CIN2/
                patient_id/
                    image.jpg
            CIN3/
                patient_id/
                    image.jpg

    Each immediate subfolder inside a class directory
    represents one patient.
    """

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE_DIR}"
        )

    patients = defaultdict(list)

    print("\nChecking merged dataset...")
    print(f"Source: {SOURCE_DIR}")

    for class_name in CLASSES:
        class_dir = SOURCE_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        # Images directly inside a class folder have no
        # explicit patient folder and cannot be safely split.
        direct_images = [
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if direct_images:
            raise RuntimeError(
                f"\nFound {len(direct_images)} image(s) directly "
                f"inside:\n{class_dir}\n\n"
                "Move them into verified patient folders before "
                "splitting. Patient IDs must not be guessed."
            )

        patient_dirs = sorted(
            path for path in class_dir.iterdir()
            if path.is_dir()
            and path.name != "reports"
        )

        print(f"\n{class_name}: {len(patient_dirs)} patient folders")

        for patient_dir in patient_dirs:
            images = get_images(patient_dir)

            if not images:
                print(
                    f"WARNING: No images found in "
                    f"{patient_dir.name}"
                )
                continue

            # Each patient is identified by its class and
            # patient-folder name in the merged dataset.
            patient_key = (class_name, patient_dir.name)

            patients[class_name].append({
                "patient_key": patient_key,
                "patient_id": patient_dir.name,
                "class_name": class_name,
                "source_dir": patient_dir,
                "image_count": len(images),
            })

    total_patients = sum(len(v) for v in patients.values())
    total_images = sum(
        patient["image_count"]
        for group in patients.values()
        for patient in group
    )

    if total_patients == 0:
        raise RuntimeError(
            "No patient folders containing images were found."
        )

    print("\n" + "=" * 65)
    print("SOURCE DATASET SUMMARY")
    print("=" * 65)

    for class_name in CLASSES:
        group = patients[class_name]
        image_count = sum(p["image_count"] for p in group)

        print(
            f"{class_name:10s} | "
            f"Patients: {len(group):4d} | "
            f"Images: {image_count:5d}"
        )

    print("-" * 65)
    print(f"Total patients: {total_patients}")
    print(f"Total images:   {total_images}")

    return patients


# ============================================================
# PATIENT-WISE SPLITTING
# ============================================================

def assign_patients(patients):
    """
    Assign whole patient folders to Train, Validation and Test.

    Assignment is performed separately for each class, aiming
    to distribute image counts approximately according to
    the requested ratios.

    No images are split individually.
    """

    rng = random.Random(RANDOM_SEED)

    assignments = {}
    split_image_counts = {
        split: defaultdict(int)
        for split in SPLIT_RATIOS
    }

    split_patient_counts = {
        split: defaultdict(int)
        for split in SPLIT_RATIOS
    }

    print("\n" + "=" * 65)
    print("ASSIGNING PATIENTS TO SPLITS")
    print("=" * 65)

    for class_name in CLASSES:
        class_patients = list(patients[class_name])

        if not class_patients:
            print(
                f"WARNING: No patients available for {class_name}."
            )
            continue

        # Shuffle first so equal-sized groups do not always
        # receive the same deterministic ordering.
        rng.shuffle(class_patients)

        # Assign larger patient groups first. This helps avoid
        # a few large groups distorting the image proportions.
        class_patients.sort(
            key=lambda patient: patient["image_count"],
            reverse=True
        )

        class_total_images = sum(
            patient["image_count"]
            for patient in class_patients
        )

        target_images = {
            split: class_total_images * ratio
            for split, ratio in SPLIT_RATIOS.items()
        }

        current_images = {
            split: 0
            for split in SPLIT_RATIOS
        }

        current_patients = {
            split: 0
            for split in SPLIT_RATIOS
        }

        for patient in class_patients:
            image_count = patient["image_count"]

            # Prefer the split that is furthest below its
            # target image count, relative to its target size.
            scores = {}

            for split, ratio in SPLIT_RATIOS.items():
                image_load = (
                    current_images[split] / ratio
                )
                patient_load = (
                    current_patients[split] / ratio
                )

                # Image balance is the primary objective.
                # Patient count is a secondary tie-breaker.
                scores[split] = (
                    image_load,
                    patient_load
                )

            best_score = min(scores.values())

            tied_splits = [
                split
                for split, score in scores.items()
                if score == best_score
            ]

            selected_split = rng.choice(tied_splits)

            assignments[patient["patient_key"]] = selected_split

            current_images[selected_split] += image_count
            current_patients[selected_split] += 1

            split_image_counts[selected_split][class_name] += (
                image_count
            )
            split_patient_counts[selected_split][class_name] += 1

        print(
            f"{class_name}: assigned "
            f"{len(class_patients)} patient(s)"
        )

    return (
        assignments,
        split_image_counts,
        split_patient_counts
    )


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():
    """
    Avoid overwriting a previous split without confirmation.
    The source dataset and old Split_Data are never deleted.
    """

    if OUTPUT_DIR.resolve() == SOURCE_DIR.resolve():
        raise RuntimeError(
            "Output directory cannot be the source directory."
        )

    if OUTPUT_DIR.exists():
        print(
            f"\nThe output folder already exists:\n{OUTPUT_DIR}"
        )

        answer = input(
            "Replace this generated split? Type YES to continue: "
        ).strip()

        if answer != "YES":
            print("Cancelled. Existing files were not changed.")
            sys.exit(0)

        # Only the specifically configured output folder
        # is removed after explicit confirmation.
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_RATIOS:
        for class_name in CLASSES:
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# COPY WHOLE PATIENT FOLDERS
# ============================================================

def copy_patient_folders(patients, assignments):
    """
    Copy each patient folder into exactly one split and class.

    The patient folder name is preserved. Original image files
    are copied without augmentation or image transformations.
    """

    print("\n" + "=" * 65)
    print("COPYING PATIENT FOLDERS")
    print("=" * 65)

    copied_patients = set()
    copied_images = 0

    for class_name in CLASSES:
        for patient in patients[class_name]:
            patient_key = patient["patient_key"]
            split = assignments[patient_key]

            source_patient_dir = patient["source_dir"]

            destination_patient_dir = (
                OUTPUT_DIR
                / split
                / class_name
                / patient["patient_id"]
            )

            if patient_key in copied_patients:
                raise RuntimeError(
                    f"Patient assigned more than once: {patient_key}"
                )

            if destination_patient_dir.exists():
                raise FileExistsError(
                    f"Destination already exists:\n"
                    f"{destination_patient_dir}"
                )

            shutil.copytree(
                source_patient_dir,
                destination_patient_dir
            )

            copied_patients.add(patient_key)
            copied_images += patient["image_count"]

            print(
                f"{split:10s} | "
                f"{class_name:7s} | "
                f"{patient['patient_id']} | "
                f"{patient['image_count']} image(s)"
            )

    print(f"\nPatient folders copied: {len(copied_patients)}")
    print(f"Images copied: {copied_images}")

    return copied_images


# ============================================================
# SAVE ASSIGNMENT REPORT
# ============================================================

def save_assignment_report(patients, assignments):
    report_path = OUTPUT_DIR / "split_assignments.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "patient_id",
                "split",
                "image_count",
                "source_folder",
                "destination_folder",
            ]
        )

        writer.writeheader()

        for class_name in CLASSES:
            for patient in patients[class_name]:
                split = assignments[patient["patient_key"]]

                writer.writerow({
                    "class": class_name,
                    "patient_id": patient["patient_id"],
                    "split": split,
                    "image_count": patient["image_count"],
                    "source_folder": str(patient["source_dir"]),
                    "destination_folder": str(
                        OUTPUT_DIR
                        / split
                        / class_name
                        / patient["patient_id"]
                    ),
                })

    print(f"Assignment report: {report_path}")


# ============================================================
# SAVE SPLIT SUMMARY
# ============================================================

def save_split_summary(patients, assignments):
    report_path = OUTPUT_DIR / "split_summary.csv"

    summary = {
        split: {
            class_name: {
                "patients": 0,
                "images": 0
            }
            for class_name in CLASSES
        }
        for split in SPLIT_RATIOS
    }

    for class_name in CLASSES:
        for patient in patients[class_name]:
            split = assignments[patient["patient_key"]]

            summary[split][class_name]["patients"] += 1
            summary[split][class_name]["images"] += (
                patient["image_count"]
            )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "class",
                "patients",
                "images",
                "image_percentage_of_split",
            ]
        )

        writer.writeheader()

        for split in SPLIT_RATIOS:
            total_split_images = sum(
                summary[split][class_name]["images"]
                for class_name in CLASSES
            )

            for class_name in CLASSES:
                patients_count = summary[split][class_name]["patients"]
                images_count = summary[split][class_name]["images"]

                percentage = (
                    100 * images_count / total_split_images
                    if total_split_images
                    else 0
                )

                writer.writerow({
                    "split": split,
                    "class": class_name,
                    "patients": patients_count,
                    "images": images_count,
                    "image_percentage_of_split": round(
                        percentage, 2
                    ),
                })

    print(f"Split summary: {report_path}")

    print("\n" + "=" * 80)
    print("FINAL SPLIT SUMMARY")
    print("=" * 80)

    for split in SPLIT_RATIOS:
        total_patients = sum(
            summary[split][class_name]["patients"]
            for class_name in CLASSES
        )

        total_images = sum(
            summary[split][class_name]["images"]
            for class_name in CLASSES
        )

        print(f"\n{split.upper()}")
        print(f"Total patients: {total_patients}")
        print(f"Total images:   {total_images}")

        for class_name in CLASSES:
            values = summary[split][class_name]

            print(
                f"  {class_name:8s} | "
                f"Patients: {values['patients']:4d} | "
                f"Images: {values['images']:5d}"
            )


# ============================================================
# VERIFY PATIENT SEPARATION
# ============================================================

def verify_patient_separation(assignments):
    """
    Verify that every recorded patient group has exactly
    one assigned split.
    """

    patient_splits = defaultdict(set)

    for patient_key, split in assignments.items():
        patient_splits[patient_key].add(split)

    leaked_patients = {
        patient_key: splits
        for patient_key, splits in patient_splits.items()
        if len(splits) > 1
    }

    if leaked_patients:
        print("\nERROR: Patient leakage detected!")

        for patient_key, splits in leaked_patients.items():
            print(patient_key, splits)

        raise RuntimeError(
            "Patient separation verification failed."
        )

    print(
        "\nPASS: No recorded patient group was assigned "
        "to more than one split."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("PATIENT-WISE DATASET SPLITTING")
    print("=" * 80)

    print("\nSplit ratios:")
    for split, ratio in SPLIT_RATIOS.items():
        print(f"  {split}: {ratio:.0%}")

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Source:      {SOURCE_DIR}")
    print(f"Destination: {OUTPUT_DIR}")

    if not abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-9:
        raise ValueError("Split ratios must add up to 1.0.")

    patients = discover_patients()

    (
        assignments,
        _,
        _
    ) = assign_patients(patients)

    # Ask before replacing a previous generated split.
    prepare_output_directory()

    copy_patient_folders(patients, assignments)

    verify_patient_separation(assignments)

    save_assignment_report(patients, assignments)
    save_split_summary(patients, assignments)

    print("\n" + "=" * 80)
    print("PATIENT-WISE SPLITTING COMPLETE")
    print("=" * 80)

    print(f"\nNew dataset: {OUTPUT_DIR}")

    print("\nReports created:")
    print(f"  {OUTPUT_DIR / 'split_assignments.csv'}")
    print(f"  {OUTPUT_DIR / 'split_summary.csv'}")

    print("\nYour original merged_dataset was not modified.")
    print("Your previous Split_Data folder was not modified.")

    print(
        "\nIMPORTANT: This split preserves the patient folders "
        "as identified in the merged dataset. If one real patient "
        "has different IDs across classes, those IDs must be "
        "reconciled before claiming person-level separation."
    )


if __name__ == "__main__":
    main()
