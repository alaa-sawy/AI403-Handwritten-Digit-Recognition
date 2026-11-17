"""UNSEAL OFFICIAL TEST ONLY AFTER all model/hyperparameter decisions are frozen.

Explicit one-time final evaluation command:
    python data/prepare_final_test.py --confirm-final

Never run during hyperparameter tuning. Raw archive remains unchanged.
"""

import argparse
import csv
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
MANIFEST = ROOT / "data" / "splits" / "test.csv"
IMAGES = ROOT / "data" / "final_test_images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final:
        parser.error("TEST is sealed; pass --confirm-final only after freezing all choices")
    if MANIFEST.exists():
        raise FileExistsError("TEST is already unsealed. Refusing to overwrite test.csv")

    records = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for source, letter in (("CAR-A", "a"), ("CAR-B", "b")):
            gt_path = f"ORAND-CAR-2014/{source}/{letter}_test_gt.txt"
            label_file = archive.extractfile(gt_path)
            if label_file is None:
                raise FileNotFoundError(gt_path)
            for line in label_file.read().decode("utf-8-sig").splitlines():
                filename, label = line.strip().split("\t")
                if Path(filename).name != filename or not label.isascii() or not label.isdigit():
                    raise ValueError(f"Bad test record: {line}")
                image_path = f"ORAND-CAR-2014/{source}/{letter}_test_images/{filename}"
                records.append({"source": source, "filename": filename,
                                "label": label, "archive_path": image_path})
        if len(records) != 6710:
            raise ValueError(f"Expected 6710 final TEST records; got {len(records)}")
        for row in records:
            destination = IMAGES / row["source"] / row["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(row["archive_path"])
            if src is None:
                raise FileNotFoundError(row["archive_path"])
            with src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "filename", "label", "archive_path"])
        writer.writeheader()
        writer.writerows(records)
    print("UNSEALED: 6710 final TEST images and labels. Do not tune models using TEST!")


if __name__ == "__main__":
    main()
