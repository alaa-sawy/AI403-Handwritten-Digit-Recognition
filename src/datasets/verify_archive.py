
from pathlib import Path
import tarfile

# 1. Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Locate the original dataset
archive_path = (
    PROJECT_ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
)

# 3. Check if the archive exists
if not archive_path.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {archive_path}"
    )

print("Dataset found!")
print("Archive size:", round(archive_path.stat().st_size / 1e6, 2), "MB")

# 4. Read the archive without extracting it
with tarfile.open(archive_path, "r:gz") as archive:
    files = [
        member.name
        for member in archive
        if member.isfile()
    ]

# 5. Count images in each official split
splits = {
    "CAR-A Train": "CAR-A/a_train_images/",
    "CAR-A Test": "CAR-A/a_test_images/",
    "CAR-B Train": "CAR-B/b_train_images/",
    "CAR-B Test": "CAR-B/b_test_images/",
}

image_extensions = (".png", ".jpg", ".jpeg")

for split_name, folder_name in splits.items():
    count = sum(
        folder_name in name
        and name.lower().endswith(image_extensions)
        for name in files
    )

    print(f"{split_name}: {count} images")

# 6. Find Ground Truth files
gt_files = [
    name for name in files
    if name.lower().endswith("_gt.txt")
]

print("\nGround Truth files:")

for name in gt_files:
    print(name)

# 7. Final verification
total_images = sum(
    name.lower().endswith(image_extensions)
    for name in files
)

if total_images == 11719 and len(gt_files) == 4:
    print("\nPASS: Archive structure verified!")
else:
    print("\nWARNING: Check archive contents!")
