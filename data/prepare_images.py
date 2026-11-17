"""M1 - Part 5: Copy TRAIN/VAL images from ORAND into a local cache.

Run once from the AI403 project folder:
    python data/prepare_images.py

Only the frozen train.csv and val.csv are used. Official TEST is untouched.
"""

import csv
import shutil
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"


def prepare_images():
    # Get image names from our fixed TRAIN and VAL splits only.
    images_to_copy = {}

    for split in ("train", "val"):
        csv_path = SPLITS_DIR / f"{split}.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                source = row["source"]
                filename = row["filename"]
                archive_name = row["archive_path"]

                if source not in ("CAR-A", "CAR-B"):
                    raise ValueError(f"Unknown source: {source}")
                if Path(filename).name != filename:
                    raise ValueError(f"Invalid filename: {filename}")
                if f"/{source}/" not in archive_name or "_train_images/" not in archive_name:
                    raise ValueError(f"Not a training image: {archive_name}")

                output_path = IMAGES_DIR / source / filename
                if not output_path.exists():
                    images_to_copy[archive_name] = output_path

    print("Images to prepare:", len(images_to_copy))

    if images_to_copy:
        # Stream through the .tar.gz just once (much faster for later training).
        with tarfile.open(ARCHIVE_PATH, "r|gz") as archive:
            copied = 0
            for member in archive:
                if member.name not in images_to_copy:
                    continue
                if not member.isfile():
                    raise ValueError(f"Not a file: {member.name}")

                output_path = images_to_copy.pop(member.name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image_file = archive.extractfile(member)
                if image_file is None:
                    raise ValueError(f"Cannot read: {member.name}")

                with image_file, output_path.open("wb") as saved_file:
                    shutil.copyfileobj(image_file, saved_file)

                copied += 1
                if copied % 1000 == 0:
                    print("Prepared:", copied)

        if images_to_copy:
            raise FileNotFoundError(
                f"Missing {len(images_to_copy)} images from the archive"
            )

    print("PASS: TRAIN/VAL images are ready in data/processed/images/")
    print("Official TEST images and labels were not copied.")


if __name__ == "__main__":
    prepare_images()
