
from pathlib import Path
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
)

TRAIN_SETS = {
    "CAR-A": ("CAR-A/a_train_gt.txt", "CAR-A/a_train_images/"),
    "CAR-B": ("CAR-B/b_train_gt.txt", "CAR-B/b_train_images/"),
}

with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:

    archive_names = {
        member.name
        for member in archive
        if member.isfile()
    }

    total_records = 0

    for set_name, (gt_rel, image_rel) in TRAIN_SETS.items():

        gt_path = f"ORAND-CAR-2014/{gt_rel}"
        gt_file = archive.extractfile(gt_path)

        if gt_file is None:
            raise FileNotFoundError(gt_path)

        lines = gt_file.read().decode("utf-8-sig").splitlines()

        image_names = [
            name for name in archive_names
            if name.startswith(f"ORAND-CAR-2014/{image_rel}")
            and name.lower().endswith(".png")
        ]

        records = []
        seen_files = set()
        invalid_labels = 0
        missing_images = 0
        malformed_lines = 0

        for line in lines:
            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                malformed_lines += 1
                continue

            filename, label = parts[0].strip(), parts[1].strip()

            if not label.isascii() or not label.isdigit():
                invalid_labels += 1

            image_path = f"ORAND-CAR-2014/{image_rel}{filename}"

            if image_path not in archive_names:
                missing_images += 1

            if filename in seen_files:
                raise ValueError(f"Duplicate filename: {filename}")

            seen_files.add(filename)
            records.append((filename, label))

        print(f"\n{set_name}")
        print("Training images:", len(image_names))
        print("Ground Truth records:", len(records))
        print("Invalid numeric labels:", invalid_labels)
        print("Missing labeled images:", missing_images)
        print("Malformed lines:", malformed_lines)
        print("Unique digit strings:", len(set(label for _, label in records)))
        print("First 3 labeled samples:", records[:3])

        assert len(image_names) == len(records), "Image/label count mismatch"
        assert invalid_labels == missing_images == malformed_lines == 0

        total_records += len(records)

print(f"\nPASS: {total_records} training images have numeric labels and matching filenames.")
print("Official test labels were NOT inspected.")
