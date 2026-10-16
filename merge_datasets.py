from pathlib import Path
import zipfile
import shutil
import hashlib
import csv
import re
from collections import defaultdict

# CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "original_datasets"

ZIP_FILES = [
    SOURCE_DIR / "archive (10).zip",
    SOURCE_DIR / "archive (11).zip",
]

OUTPUT_DIR = PROJECT_ROOT / "merged_dataset"

EXTRACT_DIR = OUTPUT_DIR / "_extracted"

REPORT_DIR = OUTPUT_DIR / "reports"


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ORIGINAL IMAGE FILTER

AUGMENTATION_KEYWORDS = {
    "flipped",
    "flip",
    "rotated",
    "rotate",
    "noisy",
    "noise",
    "augmented",
    "augmentation",
    "crop",
    "cropped",
    "resize",
    "resized",
}


def is_image(path: Path) -> bool:
    """Return True if file is a supported image."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_augmented_image(path: Path) -> bool:
    """
    Detect obvious augmented/processed images from filename or path.
    """

    text = path.as_posix().lower()

    for keyword in AUGMENTATION_KEYWORDS:
        if keyword in text:
            return True

    return False


# HASHING

def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of an image."""

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ZIP EXTRACTION

def safe_extract(zip_path: Path, destination: Path):
    """
    Safely extract ZIP without allowing files to escape
    the destination directory.
    """

    print(f"\nExtracting:")
    print(f"  {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        for member in zip_ref.infolist():

            member_path = destination / member.filename

            if not str(member_path.resolve()).startswith(
                str(destination.resolve())
            ):
                raise RuntimeError(
                    f"Unsafe ZIP entry detected: {member.filename}"
                )

        zip_ref.extractall(destination)

    print("Extraction complete.")


# FIND DIRECTORIES

def find_directory(root: Path, directory_name: str):
    """
    Find a directory by name anywhere under root.
    """

    matches = []

    for path in root.rglob("*"):

        if path.is_dir() and path.name.lower() == directory_name.lower():
            matches.append(path)

    return matches


def find_class_directories(root: Path):
    """
    Find CIN1, CIN2 and CIN3 directories.
    """

    result = {}

    for class_name in ["CIN1", "CIN2", "CIN3"]:

        matches = find_directory(root, class_name)

        if len(matches) == 0:
            print(f"WARNING: Could not find {class_name}")

        elif len(matches) > 1:

            print(
                f"WARNING: Found multiple {class_name} folders:"
            )

            for match in matches:
                print(f"   {match}")

            # Use the first one for now
            result[class_name] = matches[0]

        else:
            result[class_name] = matches[0]

    return result


def find_normal_directory(root: Path):
    """
    Find the 'Negative (normal)' directory.
    """

    candidates = []

    for path in root.rglob("*"):

        if not path.is_dir():
            continue

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            path.name.lower()
        )

        if normalized in {
            "negativenormal",
            "normal",
        }:
            candidates.append(path)

    if not candidates:
        return None

    if len(candidates) > 1:

        print("\nWARNING: Multiple possible normal folders found:")

        for c in candidates:
            print(f"   {c}")

        print("\nUsing:")
        print(candidates[0])

    return candidates[0]


# PATIENT ID EXTRACTION

def extract_malhari_patient_id(image_path: Path, class_dir: Path):
   
    try:

        relative = image_path.relative_to(class_dir)

        parts = relative.parts

        if len(parts) >= 2:

            patient_folder = parts[0]

            return patient_folder

    except ValueError:
        pass

    return None


def extract_normal_patient_id(image_path: Path):
    
    stem = image_path.stem

    # Remove obvious augmentation suffixes
    first_token = stem.split("_")[0].strip()

    if not first_token:
        return None

    return first_token

# UNIQUE PATIENT IDs

def create_unique_patient_id(class_name: str, source_patient_id: str):
    
    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        source_patient_id
    ).strip("_")

    class_clean = class_name.upper()

    return f"{class_clean}_{cleaned}"


# COPY IMAGE

def copy_image(
    source: Path,
    destination: Path
):
    """
    Copy image while preserving original filename.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )


# MAIN

def main():

    print("=" * 70)
    print("CERVICAL CANCER DATASET MERGING")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nSource directory:")
    print(SOURCE_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


    # CHECK ZIP FILES
    

    missing_zips = [
        zip_path
        for zip_path in ZIP_FILES
        if not zip_path.exists()
    ]

    if missing_zips:

        print("\nERROR: The following ZIP files were not found:")

        for path in missing_zips:
            print(f"   {path}")

        print(
            "\nMake sure the ZIP files are inside "
            "'original_datasets'."
        )

        return

   
    # CREATE OUTPUT DIRECTORIES
    

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    
    # CLEAN PREVIOUS EXTRACTION
    

    if EXTRACT_DIR.exists():

        print("\nRemoving previous temporary extraction...")

        shutil.rmtree(EXTRACT_DIR)

    EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # EXTRACT ZIP FILES
    

    extracted_roots = []

    for index, zip_path in enumerate(ZIP_FILES, start=1):

        extraction_path = (
            EXTRACT_DIR / f"source_{index}"
        )

        safe_extract(
            zip_path,
            extraction_path
        )

        extracted_roots.append(extraction_path)

    
    # FIND DATASETS
    

    print("\nSearching extracted datasets...")

    malhari_classes = {}
    normal_directory = None

    for root in extracted_roots:

        classes = find_class_directories(root)

        for class_name, directory in classes.items():

            if class_name not in malhari_classes:

                malhari_classes[class_name] = directory

        normal = find_normal_directory(root)

        if normal is not None:

            normal_directory = normal

    print("\nDetected CIN directories:")

    for class_name, directory in malhari_classes.items():

        print(f"  {class_name}: {directory}")

    print("\nDetected Normal directory:")

    print(f"  {normal_directory}")

    if not normal_directory:

        print(
            "\nERROR: Could not find the Negative (normal) "
            "directory."
        )

        print(
            "Please check the extracted Dataset Colposcopy "
            "folder structure."
        )

        return

    # --------------------------------------------------------
    # CREATE CLASS DIRECTORIES
    # --------------------------------------------------------

    for class_name in [
        "Normal",
        "CIN1",
        "CIN2",
        "CIN3",
    ]:

        (
            OUTPUT_DIR / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    
    # REPORT DATA
    

    report_rows = []

    duplicate_rows = []

    patient_summary = defaultdict(
        lambda: {
            "class": None,
            "source_patient_id": None,
            "image_count": 0,
        }
    )

    global_hashes = {}

    total_images_found = 0
    total_images_copied = 0
    total_augmented_skipped = 0
    total_duplicates_skipped = 0

    # PROCESS CIN1 / CIN2 / CIN3


    print("\n")
    print("=" * 70)
    print("PROCESSING CIN DATA")
    print("=" * 70)

    for class_name in [
        "CIN1",
        "CIN2",
        "CIN3",
    ]:

        class_dir = malhari_classes.get(class_name)

        if class_dir is None:
            continue

        print(f"\nProcessing {class_name}:")
        print(class_dir)

        images = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and is_image(p)
        ]

        print(
            f"Images found: {len(images)}"
        )

        for image_path in images:

            total_images_found += 1

            
            # Exclude augmented images
            

            if is_augmented_image(image_path):

                total_augmented_skipped += 1

                report_rows.append({
                    "class": class_name,
                    "source": "Malhari",
                    "source_patient_id": "",
                    "new_patient_id": "",
                    "source_file": str(image_path),
                    "destination_file": "",
                    "sha256": "",
                    "status": "SKIPPED_AUGMENTED",
                })

                continue

            
            # Patient ID
            

            source_patient_id = extract_malhari_patient_id(
                image_path,
                class_dir
            )

            if source_patient_id is None:

                print(
                    "\nWARNING: Could not determine patient "
                    f"for:\n{image_path}"
                )

                report_rows.append({
                    "class": class_name,
                    "source": "Malhari",
                    "source_patient_id": "",
                    "new_patient_id": "",
                    "source_file": str(image_path),
                    "destination_file": "",
                    "sha256": "",
                    "status": "REVIEW_PATIENT_ID",
                })

                continue

            new_patient_id = create_unique_patient_id(
                class_name,
                source_patient_id
            )

            
            # Hash
            

            file_hash = calculate_sha256(
                image_path
            )

            
            # Duplicate check
            

            if file_hash in global_hashes:

                previous = global_hashes[file_hash]

                total_duplicates_skipped += 1

                duplicate_rows.append({
                    "sha256": file_hash,
                    "duplicate_file": str(image_path),
                    "original_file": str(
                        previous["source_file"]
                    ),
                    "duplicate_class": class_name,
                    "original_class": previous["class"],
                })

                report_rows.append({
                    "class": class_name,
                    "source": "Malhari",
                    "source_patient_id": source_patient_id,
                    "new_patient_id": new_patient_id,
                    "source_file": str(image_path),
                    "destination_file": "",
                    "sha256": file_hash,
                    "status": "SKIPPED_DUPLICATE",
                })

                continue

            global_hashes[file_hash] = {
                "source_file": image_path,
                "class": class_name,
            }

            
            # Destination
            

            destination_dir = (
                OUTPUT_DIR
                / class_name
                / new_patient_id
            )

            destination_path = (
                destination_dir
                / image_path.name
            )

            copy_image(
                image_path,
                destination_path
            )

            total_images_copied += 1

            patient_summary[new_patient_id][
                "class"
            ] = class_name

            patient_summary[new_patient_id][
                "source_patient_id"
            ] = source_patient_id

            patient_summary[new_patient_id][
                "image_count"
            ] += 1

            report_rows.append({
                "class": class_name,
                "source": "Malhari",
                "source_patient_id": source_patient_id,
                "new_patient_id": new_patient_id,
                "source_file": str(image_path),
                "destination_file": str(destination_path),
                "sha256": file_hash,
                "status": "COPIED",
            })

    # PROCESS NORMAL DATA
    #

    print("\n")
    print("=" * 70)
    print("PROCESSING NORMAL DATA")
    print("=" * 70)

    normal_images = [
        p
        for p in normal_directory.rglob("*")
        if p.is_file() and is_image(p)
    ]

    print(
        f"\nNormal images found: {len(normal_images)}"
    )

    for image_path in normal_images:

        total_images_found += 1

        
        # Exclude augmented images
        

        if is_augmented_image(image_path):

            total_augmented_skipped += 1

            report_rows.append({
                "class": "Normal",
                "source": "Dataset Colposcopy",
                "source_patient_id": "",
                "new_patient_id": "",
                "source_file": str(image_path),
                "destination_file": "",
                "sha256": "",
                "status": "SKIPPED_AUGMENTED",
            })

            continue

       
        # Extract patient ID from filename
        

        source_patient_id = extract_normal_patient_id(
            image_path
        )

        if source_patient_id is None:

            print(
                "\nWARNING: Could not determine patient "
                f"for:\n{image_path}"
            )

            report_rows.append({
                "class": "Normal",
                "source": "Dataset Colposcopy",
                "source_patient_id": "",
                "new_patient_id": "",
                "source_file": str(image_path),
                "destination_file": "",
                "sha256": "",
                "status": "REVIEW_PATIENT_ID",
            })

            continue

        new_patient_id = create_unique_patient_id(
            "Normal",
            source_patient_id
        )

        
        # Hash
       

        file_hash = calculate_sha256(
            image_path
        )

       
        # Duplicate check
       

        if file_hash in global_hashes:

            previous = global_hashes[file_hash]

            total_duplicates_skipped += 1

            duplicate_rows.append({
                "sha256": file_hash,
                "duplicate_file": str(image_path),
                "original_file": str(
                    previous["source_file"]
                ),
                "duplicate_class": "Normal",
                "original_class": previous["class"],
            })

            report_rows.append({
                "class": "Normal",
                "source": "Dataset Colposcopy",
                "source_patient_id": source_patient_id,
                "new_patient_id": new_patient_id,
                "source_file": str(image_path),
                "destination_file": "",
                "sha256": file_hash,
                "status": "SKIPPED_DUPLICATE",
            })

            continue

        global_hashes[file_hash] = {
            "source_file": image_path,
            "class": "Normal",
        }

        
        # Destination
       

        destination_dir = (
            OUTPUT_DIR
            / "Normal"
            / new_patient_id
        )

        destination_path = (
            destination_dir
            / image_path.name
        )

        copy_image(
            image_path,
            destination_path
        )

        total_images_copied += 1

        patient_summary[new_patient_id][
            "class"
        ] = "Normal"

        patient_summary[new_patient_id][
            "source_patient_id"
        ] = source_patient_id

        patient_summary[new_patient_id][
            "image_count"
        ] += 1

        report_rows.append({
            "class": "Normal",
            "source": "Dataset Colposcopy",
            "source_patient_id": source_patient_id,
            "new_patient_id": new_patient_id,
            "source_file": str(image_path),
            "destination_file": str(destination_path),
            "sha256": file_hash,
            "status": "COPIED",
        })

    
    # SAVE IMAGE AUDIT REPORT
    

    audit_file = (
        REPORT_DIR / "image_audit.csv"
    )

    with audit_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "class",
            "source",
            "source_patient_id",
            "new_patient_id",
            "source_file",
            "destination_file",
            "sha256",
            "status",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            report_rows
        )

    
    # SAVE DUPLICATE REPORT
    

    duplicate_file = (
        REPORT_DIR / "duplicates.csv"
    )

    with duplicate_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "sha256",
            "duplicate_file",
            "original_file",
            "duplicate_class",
            "original_class",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            duplicate_rows
        )

    
    # SAVE PATIENT SUMMARY
    

    patient_file = (
        REPORT_DIR / "patient_summary.csv"
    )

    with patient_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "new_patient_id",
            "class",
            "source_patient_id",
            "image_count",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for patient_id, info in sorted(
            patient_summary.items()
        ):

            writer.writerow({
                "new_patient_id": patient_id,
                "class": info["class"],
                "source_patient_id": info[
                    "source_patient_id"
                ],
                "image_count": info[
                    "image_count"
                ],
            })

    
    # FINAL SUMMARY
    

    class_image_counts = defaultdict(int)
    class_patient_counts = defaultdict(set)

    for patient_id, info in patient_summary.items():

        class_name = info["class"]

        class_image_counts[class_name] += (
            info["image_count"]
        )

        class_patient_counts[class_name].add(
            patient_id
        )

    print("\n")
    print("=" * 70)
    print("MERGING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal image files examined: "
        f"{total_images_found}"
    )

    print(
        f"Original images copied: "
        f"{total_images_copied}"
    )

    print(
        f"Augmented/processed images skipped: "
        f"{total_augmented_skipped}"
    )

    print(
        f"Exact duplicates skipped: "
        f"{total_duplicates_skipped}"
    )

    print("\nFinal dataset:")

    for class_name in [
        "Normal",
        "CIN1",
        "CIN2",
        "CIN3",
    ]:

        print(
            f"  {class_name}: "
            f"{class_image_counts[class_name]} images | "
            f"{len(class_patient_counts[class_name])} patients"
        )

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nReports:")

    print(f"  {audit_file}")
    print(f"  {duplicate_file}")
    print(f"  {patient_file}")

    print("\nIMPORTANT:")
    print(
        "The original ZIP files were not modified."
    )

    print(
        "The '_extracted' folder is temporary and can "
        "be deleted after verification."
    )



if __name__ == "__main__":
    main()