import zipfile
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "original_datasets"

zip_files = list(DATASET_DIR.glob("*.zip"))

print("=" * 80)
print("RAW DATASET INSPECTION")
print("=" * 80)

if not zip_files:
    print("No ZIP files found.")
    raise SystemExit

for zip_path in zip_files:

    print("\n" + "=" * 80)
    print(f"DATASET: {zip_path.name}")
    print("=" * 80)

    with zipfile.ZipFile(zip_path, "r") as z:

        files = [
            name for name in z.namelist()
            if not name.endswith("/")
        ]

        print(f"\nTotal files: {len(files)}")

       
        # Show first 50 files
        
        print("\n--- First 50 files ---")

        for name in files[:50]:
            print(name)

        
        # Find image files
        
        image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        )

        images = [
            name for name in files
            if name.lower().endswith(image_extensions)
        ]

        print(f"\nTotal image files: {len(images)}")

      
        # Top-level folders
        
        top_folders = Counter()

        for name in files:
            parts = Path(name).parts

            if len(parts) >= 1:
                top_folders[parts[0]] += 1

        print("\n--- Top-level folders ---")

        for folder, count in top_folders.most_common():
            print(f"{folder}: {count}")

        
        # Second-level folders
      
        second_folders = Counter()

        for name in files:
            parts = Path(name).parts

            if len(parts) >= 2:
                second_folders[
                    str(Path(parts[0]) / parts[1])
                ] += 1

        print("\n--- Second-level folders ---")

        for folder, count in second_folders.most_common(50):
            print(f"{folder}: {count}")

print("INSPECTION COMPLETE")
