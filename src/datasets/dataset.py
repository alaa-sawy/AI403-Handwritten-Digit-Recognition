"""M1 - Part 6.2: Dataset Loader with optional TRAIN augmentation.

From the project root, run:
    python -m src.datasets.dataset
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.augment import augment_image
from src.datasets.preprocess import (
    to_grayscale,
    resize_keep_aspect,
    normalize_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"


class DigitStringDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # Validation must remain unchanged for a fair comparison.
        if split != "train" and augment:
            raise ValueError("Augmentation is allowed for train only")

        self.augment = augment

        csv_path = SPLITS_DIR / f"{split}.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            self.samples = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = IMAGES_DIR / sample["source"] / sample["filename"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing {image_path}. Run: python data/prepare_images.py"
            )

        with Image.open(image_path) as image:
            grayscale = to_grayscale(image)

            # Random changes happen before resizing and normalization.
            if self.augment:
                grayscale = augment_image(grayscale)

            resized = resize_keep_aspect(grayscale)
            pixels = normalize_image(resized)

        # CNN input format: [channels, height, width].
        image_tensor = torch.from_numpy(pixels).unsqueeze(0)

        return {
            "image": image_tensor,
            "label": sample["label"],
            "width": image_tensor.shape[-1],
            "filename": sample["filename"],
        }


if __name__ == "__main__":
    train_plain = DigitStringDataset("train", augment=False)
    train_augmented = DigitStringDataset("train", augment=True)
    val_dataset = DigitStringDataset("val")

    original = train_plain[0]
    augmented = train_augmented[0]
    val_first = val_dataset[0]
    val_second = val_dataset[0]

    print("Train samples:", len(train_plain))
    print("Validation samples:", len(val_dataset))
    print("Train without augmentation:", original["image"].shape)
    print("Train with augmentation:", augmented["image"].shape)
    print("Label before:", original["label"])
    print("Label after:", augmented["label"])
    print("Validation unchanged:", torch.equal(val_first["image"], val_second["image"]))

    assert len(train_plain) == 4507
    assert len(val_dataset) == 501
    assert original["label"] == augmented["label"]
    assert original["image"].shape[0:2] == (1, 48)
    assert augmented["image"].shape[0:2] == (1, 48)
    assert augmented["image"].dtype == torch.float32
    assert 0 <= augmented["image"].min() <= augmented["image"].max() <= 1
    assert torch.equal(val_first["image"], val_second["image"])

    print("\nPASS: Optional TRAIN augmentation works, validation stays fixed!")
