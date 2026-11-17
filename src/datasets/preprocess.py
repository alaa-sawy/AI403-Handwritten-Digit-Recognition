"""M1 - Part 4.3: Grayscale, resize, and normalization.

Run from the AI403 project root:
    python src/datasets/preprocess.py

Requires:
    python -m pip install pillow numpy
"""

from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "audit_outputs"
TARGET_HEIGHT = 48


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale, replacing transparency with white."""
    rgba = image.convert("RGBA")
    white_background = Image.new("RGBA", rgba.size, "white")
    image_on_white = Image.alpha_composite(white_background, rgba)
    return image_on_white.convert("L")


def resize_keep_aspect(
    image: Image.Image, target_height: int = TARGET_HEIGHT
) -> Image.Image:
    """Resize to a fixed height without stretching the handwriting."""
    if target_height <= 0:
        raise ValueError("target_height must be positive")

    old_width, old_height = image.size
    new_width = max(1, round(old_width * target_height / old_height))

    return image.resize(
        (new_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )


def normalize_image(image: Image.Image) -> np.ndarray:
    """Return float32 pixel values between 0.0 and 1.0."""
    pixels = np.asarray(image, dtype=np.float32)
    return pixels / 255.0


if __name__ == "__main__":
    # This is an image from the official TRAIN set (not TEST).
    image_path = (
        "ORAND-CAR-2014/CAR-A/a_train_images/"
        "a_car_000154.png"
    )

    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE_PATH}")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        image_file = archive.extractfile(image_path)
        if image_file is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(BytesIO(image_file.read())) as original:
            grayscale = to_grayscale(original)
            resized = resize_keep_aspect(grayscale)
            normalized = normalize_image(resized)

            print("Original mode:", original.mode)
            print("Grayscale mode:", grayscale.mode)
            print("Original size:", original.size)
            print("Grayscale size:", grayscale.size)
            print("Resized size:", resized.size)
            print("Normalized shape:", normalized.shape)
            print("Normalized dtype:", normalized.dtype)
            print("Pixel range:", round(float(normalized.min()), 3),
                  "to", round(float(normalized.max()), 3))

            assert grayscale.mode == "L"
            assert grayscale.size == original.size
            assert resized.height == TARGET_HEIGHT
            assert normalized.shape == (resized.height, resized.width)
            assert normalized.dtype == np.float32
            assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Keep existing preview images unchanged on repeated runs.
            previews = {
                "sample_original.png": original,
                "sample_grayscale.png": grayscale,
                "sample_resized.png": resized,
            }
            for filename, image in previews.items():
                output_path = OUTPUT_DIR / filename
                if not output_path.exists():
                    image.save(output_path)

            np.save(OUTPUT_DIR / "sample_normalized.npy", normalized)

    print("\nPASS: Grayscale conversion successful!")
    print("PASS: Resize successful!")
    print("PASS: Normalization successful!")
