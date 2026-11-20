"""Local, checkpoint-only digit-string inference for the NeuroMotion demo.

No training, dataset TEST access, remote model downloads or checkpoint selection.
"""

from pathlib import Path

import torch
from PIL import Image

from src.ctc_utils import greedy_decode
from src.datasets.preprocess import normalize_image, resize_keep_aspect, to_grayscale
from src.models.registry import build_model

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 21, 42)

# Each entry is (config model name, frozen checkpoint path relative to project root).
MODEL_CHECKPOINTS = {
    "Classic Pretrained": (
        "classic_pretrained", "checkpoints/classic_pretrained_results/final/seed_{seed}/best.pt"
    ),
    "Advanced Full": (
        "advanced", "checkpoints/advanced_results/final/seed_{seed}/best.pt"
    ),
    "Classic Scratch": (
        "classic_scratch", "checkpoints/classic_scratch_results/final/seed_{seed}/best.pt"
    ),
    "Modern Augmentation ON": (
        "modern", "checkpoints/modern_results/final/seed_{seed}/best.pt"
    ),
    "Modern Augmentation OFF": (
        "modern", "checkpoints/modern_results/augmentation_off/seed_{seed}/best.pt"
    ),
    "Advanced No-Multiscale": (
        "advanced_no_multiscale",
        "checkpoints/advanced_results/ablation_no_multiscale/seed_{seed}/best.pt",
    ),
    "Baseline CNN": (
        "baseline", "checkpoints/baseline_results/final/seed_{seed}/best.pt"
    ),
}


def checkpoint_for(model_label: str, seed: int) -> Path:
    """Return the expected local checkpoint path; do not silently choose another."""
    if model_label not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model selection: {model_label!r}")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}, got {seed!r}")
    return ROOT / MODEL_CHECKPOINTS[model_label][1].format(seed=seed)


def load_checkpoint_model(model_name, checkpoint_path, device="cpu", model_label=None):
    """Strictly restore a previously trained model; never retrain/download weights.

    Only load checkpoints produced by this project from trusted local storage.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {path}. Copy the frozen best.pt from the team's "
            "checkpoint Drive into this exact folder. Do not retrain."
        )
    if isinstance(device, str) and device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but no CUDA GPU is available")

    # Safe mode: never execute arbitrary Python objects from a checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing config or model_state_dict")
    if config.get("model") != model_name:
        raise ValueError(
            f"Wrong model checkpoint: expected {model_name!r}, "
            f"found {config.get('model')!r}"
        )
    dropout = config.get("dropout")
    if not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("Checkpoint has an invalid dropout configuration")

    if model_label is not None:
        if model_label not in MODEL_CHECKPOINTS or MODEL_CHECKPOINTS[model_label][0] != model_name:
            raise ValueError("Model label does not match selected architecture")
        seed = config.get("seed")
        if type(seed) is not int or seed not in SEEDS:
            raise ValueError("Checkpoint has an invalid final seed")
        if path != checkpoint_for(model_label, seed).resolve():
            raise ValueError("Checkpoint file is not the requested model/seed path")
        if model_label.startswith("Modern Augmentation "):
            expected_aug = model_label.endswith("ON")
            if config.get("augmentation") is not expected_aug:
                raise ValueError("Modern checkpoint augmentation flag does not match selection")

    # The scratch and pretrained Classic models have identical *final* architecture.
    # Constructing the scratch version prevents torchvision from downloading
    # ImageNet initialization, which final weights would overwrite anyway.
    constructor_name = "classic_scratch" if model_name == "classic_pretrained" else model_name
    model = build_model(constructor_name, dropout=float(dropout))
    model.load_state_dict(state_dict, strict=True)
    model.to(target).eval()
    return model, checkpoint, target


@torch.inference_mode()
def predict_image(image: Image.Image, model, device="cpu"):
    """Return (decoded digit string, resized grayscale image, resized width)."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image is empty")
    processed = resize_keep_aspect(to_grayscale(image))
    pixels = normalize_image(processed)
    width = int(pixels.shape[1])
    if width < 4:
        raise ValueError("Image is too narrow after resizing; upload a clearer cropped image")
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)
    widths = torch.tensor([width], dtype=torch.long)  # Same valid-width contract as training.
    logits, output_lengths = model(tensor, widths)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 11:
        raise ValueError(f"Model returned invalid logits shape: {tuple(logits.shape)}")
    prediction = greedy_decode(logits, torch.as_tensor(output_lengths))[0]
    return prediction, processed, width
