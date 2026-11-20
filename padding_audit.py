"""Padding impact audit for M4 checkpoints (validation split only)."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ctc_utils import collate_batch, greedy_decode
from src.datasets.dataset import DigitStringDataset
from src.evaluate import metrics_from_predictions
from src.models.registry import build_model


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    model = build_model(config["model"], config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return checkpoint, model


@torch.no_grad()
def compare_single_vs_padded(model, dataset, device):
    """Compare one image alone and as the first item beside a wider image."""
    samples = [dataset[i] for i in range(len(dataset))]
    widest_index = max(range(len(samples)), key=lambda i: samples[i]["width"])

    # Choose a sample narrower than the widest one, if possible.
    candidates = [
        i for i, sample in enumerate(samples)
        if sample["width"] < samples[widest_index]["width"]
    ]
    if not candidates:
        raise RuntimeError("Validation set has no pair with different widths.")

    sample_index = candidates[0]
    target = samples[sample_index]
    wider = samples[widest_index]

    alone_batch = collate_batch([target])
    mixed_batch = collate_batch([target, wider])

    alone_logits, alone_lengths = model(
        alone_batch["images"].to(device),
        alone_batch["widths"],
    )
    mixed_logits, mixed_lengths = model(
        mixed_batch["images"].to(device),
        mixed_batch["widths"],
    )

    valid_length = int(alone_lengths[0])
    alone_valid = alone_logits[0, :valid_length]
    mixed_valid = mixed_logits[0, :valid_length]

    max_abs_diff = (alone_valid - mixed_valid).abs().max().item()
    mean_abs_diff = (alone_valid - mixed_valid).abs().mean().item()

    alone_prediction = greedy_decode(alone_logits, alone_lengths)[0]
    mixed_prediction = greedy_decode(
        mixed_logits[:1], mixed_lengths[:1]
    )[0]

    return {
        "sample_filename": target.get("filename", ""),
        "sample_width": int(target["width"]),
        "wider_filename": wider.get("filename", ""),
        "wider_width": int(wider["width"]),
        "valid_timesteps": valid_length,
        "max_abs_logit_difference": max_abs_diff,
        "mean_abs_logit_difference": mean_abs_diff,
        "prediction_alone": alone_prediction,
        "prediction_in_wider_batch": mixed_prediction,
        "prediction_unchanged": alone_prediction == mixed_prediction,
    }


@torch.no_grad()
def evaluate_at_batch_size(model, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    records = []

    for batch in loader:
        logits, lengths = model(
            batch["images"].to(device),
            batch["widths"],
        )
        predictions = greedy_decode(logits, lengths)

        for filename, label, prediction in zip(
            batch["filenames"], batch["labels"], predictions
        ):
            records.append({
                "filename": filename,
                "label": label,
                "prediction": prediction,
            })

    summary, _, _, _ = metrics_from_predictions(records)
    return {
        "batch_size": batch_size,
        "cer": summary["cer"],
        "exact_match": summary["exact_match"],
        "sample_count": summary["sample_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more final M4 checkpoint paths.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32, 64],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DigitStringDataset("val")
    output = {
        "split": "validation",
        "device": str(device),
        "batch_sizes": args.batch_sizes,
        "checkpoints": [],
    }

    for checkpoint_arg in args.checkpoints:
        checkpoint_path = Path(checkpoint_arg).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint, model = load_model(checkpoint_path, device)

        pair_result = compare_single_vs_padded(model, dataset, device)
        batch_results = [
            evaluate_at_batch_size(model, dataset, size, device)
            for size in args.batch_sizes
        ]

        item = {
            "checkpoint": str(checkpoint_path),
            "model": checkpoint["config"]["model"],
            "single_vs_padded_pair": pair_result,
            "batch_size_metrics": batch_results,
        }
        output["checkpoints"].append(item)

        print(f"\nCheckpoint: {checkpoint_path}")
        print("Model:", item["model"])
        print("Single vs wider batch:")
        print(json.dumps(pair_result, indent=2))
        print("Validation metrics by batch size:")
        for result in batch_results:
            print(
                f"  batch={result['batch_size']:>3} | "
                f"CER={result['cer']:.6f} | "
                f"Exact={result['exact_match']:.6f}"
            )

    output_path = Path("experiments") / "padding_audit_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
