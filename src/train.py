"""AI403 shared Trainer for baseline, classic, modern and advanced OCR models.

The model contract is fixed:
    model(images [B,1,48,W], valid_widths [B])
         -> (logits [B,T,11], valid_output_lengths [B])

Direct smoke test (does not save checkpoints):
    python -m src.train --smoke
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.ctc_utils import collate_batch, ctc_forward
from src.evaluate import evaluate_model, save_evaluation
from src.models.registry import build_model
from src.datasets.dataset import DigitStringDataset

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    """Set the random state for repeatable runs on the same environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available")
    if requested not in ("cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu")
    return torch.device(requested)


def validate_config(config):
    for field in ("model", "optimizer", "learning_rate", "dropout", "batch_size", "epochs",
                  "patience", "weight_decay", "gradient_clip", "augmentation", "seed"):
        if field not in config:
            raise ValueError(f"Required config field missing: {field}")
    if not (config["learning_rate"] > 0 and 0 <= config["dropout"] < 1):
        raise ValueError("learning_rate must be positive and dropout in [0,1)")
    if config["batch_size"] < 1 or config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("batch_size, epochs and patience must be positive")
    if config["weight_decay"] < 0 or config["gradient_clip"] <= 0:
        raise ValueError("Invalid weight_decay or gradient_clip")
    if config["optimizer"].lower() not in ("adam", "adamw"):
        raise ValueError("Only Adam / AdamW are supported")
    if not isinstance(config["augmentation"], bool):
        raise ValueError("augmentation must be true or false")


def build_loaders(config, smoke=False, tiny_overfit=False):
    """CAR-A and CAR-B stay mixed in each fixed CSV manifest."""
    train_set = DigitStringDataset("train", augment=config["augmentation"] and not tiny_overfit)
    val_set = DigitStringDataset("val", augment=False)
    if smoke:
        train_set = Subset(train_set, range(min(len(train_set), 8)))
        val_set = Subset(val_set, range(min(len(val_set), 4)))
    if tiny_overfit:
        # Short, non-repeated labels make the tiny memorization gate practical
        # even for the intentionally weak CNN-only baseline.
        eligible = [i for i, row in enumerate(train_set.samples)
                    if all(a != b for a, b in zip(row["label"], row["label"][1:]))]
        eligible.sort(key=lambda i: len(train_set.samples[i]["label"]))
        if len(eligible) < 4:
            raise ValueError("Need four TRAIN images for tiny-overfit gate")
        train_set = Subset(train_set, eligible[:4])
        val_set = train_set  # intentionally test memorization, NOT generalization
    generator = torch.Generator().manual_seed(config["seed"])
    common = dict(batch_size=config["batch_size"], collate_fn=collate_batch,
                  num_workers=config.get("num_workers", 0), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


def model_parameters(model):
    return sum(param.numel() for param in model.parameters())


def split_fingerprint():
    hashes = {}
    for split in ("train", "val"):
        path = ROOT / "data" / "splits" / f"{split}.csv"
        hashes[f"{split}_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def hardware_info(device):
    info = {"device": str(device), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    else:
        info["gpu"] = None
    return info


def train_one_epoch(model, loader, loss_fn, optimizer, config, device, memory_stats=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = ctc_forward(model, batch, loss_fn, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        if memory_stats is not None:
            memory_stats["observed_rss_mb"] = max(
                memory_stats["observed_rss_mb"],
                memory_stats["process"].memory_info().rss / 1048576)
        count = len(batch["labels"])
        running_loss += loss.item() * count
        total_samples += count
    return running_loss / total_samples


def train_model(model, train_loader, val_loader, config, run_dir=None, smoke=False):
    """Shared training loop: all models use exactly this optimizer/loss/selection logic."""
    validate_config(config)
    device = get_device(config)
    model = model.to(device)
    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    optimizer_type = torch.optim.Adam if config["optimizer"].lower() == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = None
    if run_dir is not None and not smoke:
        run_dir = Path(run_dir).resolve()
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use a new run name.")
        # Keep model weights out of experiments/; save only logs/tables there.
        relative_run = run_dir.relative_to(ROOT / "experiments")
        checkpoint_path = ROOT / "checkpoints" / relative_run / "best.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
        run_dir.mkdir(parents=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_cer = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    epochs_to_run = 1 if smoke else config["epochs"]
    history = []
    memory_stats = ({"process": psutil.Process(os.getpid()), "observed_rss_mb": 0.0}
                    if psutil is not None else None)
    start = time.perf_counter()
    initial_state = [param.detach().cpu().clone() for param in model.parameters()]

    for epoch in range(1, epochs_to_run + 1):
        started_epoch = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, config, device, memory_stats)
        val = evaluate_model(model, val_loader, loss_fn, device)
        elapsed_epoch = time.perf_counter() - started_epoch
        if memory_stats is not None:
            memory_stats["observed_rss_mb"] = max(
                memory_stats["observed_rss_mb"], memory_stats["process"].memory_info().rss / 1048576)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["ctc_loss"],
                  "val_cer": val["cer"], "val_exact_match": val["exact_match"],
                  "epoch_seconds": elapsed_epoch}
        history.append(record)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
              f"val_loss={val['ctc_loss']:.4f} CER={val['cer']:.4f} "
              f"Exact={val['exact_match']:.4f} ({elapsed_epoch:.1f}s)", flush=True)

        # Primary criterion is validation CER; loss breaks ties on equal CER.
        improved = val["cer"] < best_cer - 1e-12 or (
            abs(val["cer"] - best_cer) <= 1e-12 and val["ctc_loss"] < best_loss - 1e-8)
        if improved:
            best_cer, best_loss, best_epoch = val["cer"], val["ctc_loss"], epoch
            no_improvement = 0
            if run_dir is not None and not smoke:
                torch.save({"model_state_dict": model.state_dict(),
                            "config": config, "best_epoch": epoch, "val_cer": best_cer,
                            "split_sha256": split_fingerprint()}, checkpoint_path)
        else:
            no_improvement += 1
        if config.get("tiny_overfit_gate") and val["exact_match"] >= 1.0:
            print("Tiny-overfit achieved: 4/4 labels memorized")
            break
        if not smoke and no_improvement >= config["patience"]:
            print(f"Early stopping: validation did not improve for {config['patience']} epochs")
            break

    elapsed = time.perf_counter() - start
    weights_changed = any(not torch.equal(a, b.detach().cpu())
                          for a, b in zip(initial_state, model.parameters()))
    if not weights_changed:
        raise AssertionError("Model parameters did not update: check optimizer and gradients")
    if smoke:
        print("\nPASS: mixed-width CNN + CTC + backward + validation (smoke test)")
        print("No checkpoint saved; no official TEST labels opened.")
        return {"smoke_pass": True, "weights_updated": True}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    full_val, per_digit, confusions, rows = evaluate_model(
        model, val_loader, loss_fn, device, collect_predictions=True)
    save_evaluation(run_dir / "validation", full_val, per_digit, confusions, rows)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Save compact figures alongside the important experiment logs.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [row["epoch"] for row in history]
        for filename, columns, title in (
            ("loss_curve.png", ("train_loss", "val_loss"), "CTC Loss by epoch"),
            ("cer_curve.png", ("val_cer",), "Validation CER by epoch"),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            for column in columns:
                ax.plot(epochs, [row[column] for row in history], label=column)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.grid(alpha=0.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / filename, dpi=150)
            plt.close(fig)
    except ImportError:
        print("Plotting skipped: install matplotlib for training curves")

    # Best model is selected ONLY using validation.
    result = {
        "model": config["model"], "seed": config["seed"],
        "learning_rate": config["learning_rate"], "dropout": config["dropout"],
        "augmentation": config["augmentation"],
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "val_cer": full_val["cer"], "val_exact_match": full_val["exact_match"],
        "train_seconds": elapsed,
        "inference_ms_per_image": full_val["inference_ms_per_image"],
        "parameter_count": model_parameters(model),
        "model_weights_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1048576,
        "checkpoint_size_mb": checkpoint_path.stat().st_size / 1048576,
        "checkpoint_path": str(checkpoint_path),
        "gpu_peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 1048576)
                                 if device.type == "cuda" else None,
        "cpu_process_observed_peak_rss_mb": (memory_stats["observed_rss_mb"]
                                              if memory_stats is not None else None),
        "hardware": hardware_info(device),
        "split_sha256": split_fingerprint(),
        "test_was_used": False,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Saved checkpoint:", checkpoint_path)
    print("Saved result files:", run_dir)
    return result


def run_experiment(config, run_dir=None, smoke=False, tiny_overfit=False):
    """Rebuild a NEW model for each seed/trial to keep runs independent."""
    validate_config(config)
    set_seed(config["seed"])
    train_loader, val_loader = build_loaders(config, smoke=smoke, tiny_overfit=tiny_overfit)
    model = build_model(config["model"], dropout=config["dropout"])
    return train_model(model, train_loader, val_loader, config, run_dir=run_dir, smoke=smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--augment", choices=["on", "off"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model is not None:
        config["model"] = args.model
    for name, value in (("learning_rate", args.lr), ("dropout", args.dropout),
                        ("seed", args.seed), ("epochs", args.epochs)):
        if value is not None:
            config[name] = value
    if args.augment is not None:
        config["augmentation"] = args.augment == "on"
    run_name = args.run_name or (f"{config['model']}_seed{config['seed']}_"
                                  f"lr{config['learning_rate']}_do{config['dropout']}_"
                                  f"aug{int(config['augmentation'])}")
    folder = "baseline_results" if config["model"] == "baseline" else config["model"] + "_results"
    out_dir = ROOT / "experiments" / folder / run_name
    run_experiment(config, run_dir=out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
