"""M1 - Part 6.1: Light data augmentation for TRAIN images only.

Run the preview from the AI403 project folder:
    python -m src.datasets.augment

Augmentation is applied to a grayscale PIL image BEFORE resize and normalize.
Do not apply it to validation or test images.
"""

import random
from pathlib import Path

from PIL import Image, ImageEnhance

from src.datasets.preprocess import (
    normalize_image,
    resize_keep_aspect,
    to_grayscale,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def augment_image(image: Image.Image, rng=None) -> Image.Image:
    """Apply a small random rotation and brightness change."""
    if image.mode != "L":
        raise ValueError("augment_image expects a grayscale ('L') image")

    if rng is None:
        rng = random

    # expand=True avoids cutting off digits near image borders.
    angle = rng.uniform(-2.0, 2.0)
    rotated = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=255,
    )

    brightness = rng.uniform(0.90, 1.10)
    return ImageEnhance.Brightness(rotated).enhance(brightness)


if __name__ == "__main__":
    # Example from CAR-A's official TRAIN set.
    image_path = (
        PROJECT_ROOT / "data" / "processed" / "images"
        / "CAR-A" / "a_car_000154.png"
    )
    output_dir = PROJECT_ROOT / "data" / "audit_outputs"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Training image not found: {image_path}\n"
            "First run: python data/prepare_images.py"
        )

    with Image.open(image_path) as original:
        grayscale = to_grayscale(original)

    # Fixed random seed ONLY for this preview, so it is repeatable.
    augmented = augment_image(grayscale, rng=random.Random(403))
    resized = resize_keep_aspect(augmented)
    normalized = normalize_image(resized)

    assert augmented.mode == "L"
    assert resized.height == 48
    assert normalized.shape == (48, resized.width)
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.save(output_dir / "sample_augmented.png")
    resized.save(output_dir / "sample_augmented_resized.png")

    print("Original size:", grayscale.size)
    print("Augmented size:", augmented.size)
    print("Resized size:", resized.size)
    print("Normalized shape:", normalized.shape)
    print("Label remains: 2250")
    print("Saved previews in data/audit_outputs/")
    print("\nPASS: Train augmentation preview successful!")
