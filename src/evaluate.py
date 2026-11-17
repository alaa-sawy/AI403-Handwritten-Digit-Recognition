"""Shared OCR evaluation: CER, exact match, per-digit scores and real errors.

Use after training:
    python -m src.evaluate --checkpoint checkpoints/baseline_..._best.pt
Official TEST is gated: --split test --final-test, after freezing all choices.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.ctc_utils import collate_batch, ctc_forward, greedy_decode

ROOT = Path(__file__).resolve().parents[1]
DIGITS = "0123456789"


def aligned_operations(truth: str, prediction: str):
    """Levenshtein alignment: match, substitution, deletion, insertion."""
    n, m = len(truth), len(prediction)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (truth[i - 1] != prediction[j - 1]),
            )
    operations = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (truth[i - 1] != prediction[j - 1]):
            a, b = truth[i - 1], prediction[j - 1]
            operations.append(("match" if a == b else "substitution", a, b))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            operations.append(("deletion", truth[i - 1], ""))
            i -= 1
        else:
            operations.append(("insertion", "", prediction[j - 1]))
            j -= 1
    operations.reverse()
    return operations


def source_from_filename(name):
    if name.startswith("a_car_"):
        return "CAR-A"
    if name.startswith("b_car_"):
        return "CAR-B"
    return "unknown"


def metrics_from_predictions(rows):
    """Compute corpus CER and true digit-wise precision/recall/F1 via alignment."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot evaluate empty predictions")
    counts = {d: Counter() for d in DIGITS}
    substitutions = Counter()
    grouped = defaultdict(lambda: Counter())
    total_edits = total_characters = exact = 0
    for row in rows:
        truth, prediction = row["label"], row["prediction"]
        operations = aligned_operations(truth, prediction)
        edits = sum(operation != "match" for operation, _, _ in operations)
        row["edit_distance"] = edits
        row["exact_match"] = int(truth == prediction)
        row["source"] = row.get("source") or source_from_filename(row.get("filename", ""))
        row["length"] = len(truth)
        total_edits += edits
        total_characters += len(truth)
        exact += truth == prediction
        for key in (f"source:{row['source']}", f"length:{len(truth)}"):
            grouped[key]["edits"] += edits
            grouped[key]["chars"] += len(truth)
            grouped[key]["exact"] += truth == prediction
            grouped[key]["samples"] += 1
        for operation, a, b in operations:
            if operation == "match":
                counts[a]["tp"] += 1
            elif operation == "substitution":
                counts[a]["fn"] += 1
                counts[b]["fp"] += 1
                counts[a]["substitutions"] += 1
                substitutions[(a, b)] += 1
            elif operation == "deletion":
                counts[a]["fn"] += 1
                counts[a]["deletions"] += 1
            else:
                counts[b]["fp"] += 1
                counts[b]["insertions"] += 1
    per_digit = []
    for d, v in counts.items():
        precision = v["tp"] / (v["tp"] + v["fp"]) if (v["tp"] + v["fp"]) else 0.0
        recall = v["tp"] / (v["tp"] + v["fn"]) if (v["tp"] + v["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_digit.append({
            "digit": d, "support": v["tp"] + v["fn"],
            "precision": precision, "recall": recall, "f1": f1,
            "tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
            "substitutions": v["substitutions"], "deletions": v["deletions"],
            "insertions": v["insertions"],
        })
    groups = {
        k: {"cer": v["edits"] / v["chars"] if v["chars"] else 0,
            "exact_match": v["exact"] / v["samples"], "samples": v["samples"]}
        for k, v in sorted(grouped.items())
    }
    summary = {
        "cer": total_edits / total_characters,
        "exact_match": exact / len(rows),
        "sample_count": len(rows), "total_edits": total_edits,
        "total_characters": total_characters,
        "by_source_and_length": groups,
    }
    return summary, per_digit, substitutions, rows


def save_evaluation(output_dir, summary, per_digit, substitutions, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, values, fields in [
        ("per_digit_metrics.csv", per_digit,
         ["digit", "support", "precision", "recall", "f1", "tp", "fp", "fn", "substitutions", "deletions", "insertions"]),
        ("predictions.csv", rows,
         ["filename", "source", "label", "prediction", "length", "edit_distance", "exact_match"]),
        ("errors.csv", [row for row in rows if row["edit_distance"]],
         ["filename", "source", "label", "prediction", "length", "edit_distance", "exact_match"]),
        ("substitution_confusions.csv",
         [{"true_digit": a, "predicted_digit": b, "count": n} for (a, b), n in sorted(substitutions.items())],
         ["true_digit", "predicted_digit", "count"]),
    ]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, collect_predictions=False):
    """Called by every training epoch; optionally return per-image records."""
    model.eval()
    losses = 0.0
    records = []
    samples = 0
    inference_stats = {"seconds": 0.0}
    for batch in loader:
        loss, logits, lengths = ctc_forward(model, batch, criterion, device, inference_stats)
        predictions = greedy_decode(logits, lengths)
        for filename, truth, prediction in zip(batch["filenames"], batch["labels"], predictions):
            records.append({"filename": filename, "label": truth, "prediction": prediction})
        count = len(batch["labels"])
        losses += loss.item() * count
        samples += count
    summary, per_digit, substitutions, records = metrics_from_predictions(records)
    summary["ctc_loss"] = losses / samples
    summary["inference_ms_per_image"] = 1000 * inference_stats["seconds"] / samples
    if collect_predictions:
        return summary, per_digit, substitutions, records
    return summary


class FinalTestDataset(Dataset):
    """Only used after data/prepare_final_test.py has been deliberately run."""
    def __init__(self):
        manifest = ROOT / "data" / "splits" / "test.csv"
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            self.samples = list(csv.DictReader(file))
        if len(self.samples) != 6710:
            raise ValueError("Final TEST manifest must contain 6710 labeled samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from src.datasets.preprocess import to_grayscale, resize_keep_aspect, normalize_image
        row = self.samples[index]
        path = ROOT / "data" / "final_test_images" / row["source"] / row["filename"]
        with Image.open(path) as image:
            arr = normalize_image(resize_keep_aspect(to_grayscale(image)))
        image_tensor = torch.from_numpy(arr).unsqueeze(0)
        return {"image": image_tensor, "width": image_tensor.shape[-1],
                "label": row["label"], "filename": row["filename"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-test", action="store_true",
                        help="Required to unlock the official test, after all choices are frozen")
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("TEST is sealed. Freeze model selection first, then pass --final-test")
    if args.split == "val" and args.final_test:
        parser.error("--final-test is only for split=test")
    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from src.models.registry import build_model
    from src.datasets.dataset import DigitStringDataset
    config = checkpoint["config"]
    model = build_model(config["model"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = DigitStringDataset("val") if args.split == "val" else FinalTestDataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        collate_fn=collate_batch, num_workers=0)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    summary, per_digit, substitutions, rows = evaluate_model(
        model, loader, criterion, device, collect_predictions=True)
    folder = checkpoint_path.parent / f"{checkpoint_path.stem}_{args.split}_evaluation"
    save_evaluation(folder, summary, per_digit, substitutions, rows)
    print(f"{args.split.upper()} CER: {summary['cer']:.4f} | Exact match: {summary['exact_match']:.4f}")
    print("Evaluation files:", folder)


if __name__ == "__main__":
    main()
