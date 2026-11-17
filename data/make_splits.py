"""AI403 M1 — Part 3: freeze Train/Validation without opening test labels.

Input:
  data/raw/ORAND-CAR-2014.tar.gz
  data/audit_outputs/data_audit_images.csv  (produced by audit_images.py)
Output:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test_manifest_sealed.csv (names only, no labels)
  data/audit_outputs/split_summary.txt

Use from project root:
  python data/make_splits.py
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import io
import math
import random
import tarfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "data/audit_outputs" / "data_audit_images.csv"
ARCHIVE = ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
SPLITS_DIR = ROOT / "data" / "splits"
AUDIT_OUTPUT_DIR = ROOT / "data/audit_outputs"
SEED = 403
VAL_FRACTION = 0.10
EXPECTED_ORIGINAL_TRAIN = 5009
EXPECTED_OFFICIAL_TEST = 6710
EXPECTED_DUPLICATES = 1


def full_train_path(row):
    source = row["source"]
    prefix = {"CAR-A": "a", "CAR-B": "b"}[source]
    return f"ORAND-CAR-2014/{source}/{prefix}_train_images/{row['filename']}"


def load_training_audit():
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError(f"Missing audit: {AUDIT_CSV}")
    with AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"source", "filename", "label", "sha256", "status", "width", "height"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Audit columns missing. Expected {sorted(required)}")
    if len(rows) != EXPECTED_ORIGINAL_TRAIN:
        raise ValueError(f"Expected {EXPECTED_ORIGINAL_TRAIN} training images, found {len(rows)}")
    seen_names = set()
    for row in rows:
        if row["source"] not in {"CAR-A", "CAR-B"}:
            raise ValueError("Unknown source: " + row["source"])
        key = (row["source"], row["filename"])
        if key in seen_names:
            raise ValueError(f"Duplicate source/filename in audit: {key}")
        seen_names.add(key)
        if row["status"].strip().upper() != "OK":
            raise ValueError(f"Invalid audit status: {key}")
        if not row["label"].isascii() or not row["label"].isdigit():
            raise ValueError(f"Invalid label: {key}")
        if len(row["sha256"]) != 64:
            raise ValueError(f"Bad SHA256: {key}")
        row["archive_path"] = full_train_path(row)
        row["label_length"] = str(len(row["label"]))
    return rows


def remove_exact_duplicates(rows):
    """Keep one example per SHA-256; never alter the raw archive."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)

    kept = []
    removed = []
    for digest, group in sorted(by_hash.items()):
        group = sorted(group, key=lambda r: (r["source"], r["filename"]))
        if len({r["label"] for r in group}) != 1:
            raise ValueError(f"Conflicting labels for same image SHA256: {digest}")
        kept.append(group[0])
        removed.extend(group[1:])
    if len(removed) != EXPECTED_DUPLICATES:
        raise ValueError(f"Expected {EXPECTED_DUPLICATES} duplicate image; got {len(removed)}")
    return kept, removed


def stratified_split(rows):
    """Stratify by CAR source + label length; keep tiny strata in train.

    Uses a fixed seed. Only official TRAIN is considered.
    Does not force different labels into different splits: label repeats are normal.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["source"], len(r["label"]))].append(r)

    target_val = round(len(rows) * VAL_FRACTION)
    quotas = {}
    ideal = {}
    for key, bucket in buckets.items():
        ideal[key] = len(bucket) * VAL_FRACTION
        quotas[key] = min(math.floor(ideal[key]), max(0, len(bucket) - 1))

    # Largest remaining fractional quota gets the next validation sample.
    while sum(quotas.values()) < target_val:
        eligible = [k for k, bucket in buckets.items() if quotas[k] < len(bucket) - 1]
        if not eligible:
            raise ValueError("Not enough eligible records for validation fraction")
        best = sorted(eligible, key=lambda k: (-(ideal[k] - quotas[k]), k))[0]
        quotas[best] += 1

    rng = random.Random(SEED)
    train, val = [], []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda r: (r["source"], r["filename"]))
        rng.shuffle(bucket)
        val.extend(bucket[:quotas[key]])
        train.extend(bucket[quotas[key]:])

    train.sort(key=lambda r: (r["source"], r["filename"]))
    val.sort(key=lambda r: (r["source"], r["filename"]))
    return train, val


def get_test_filename_manifest():
    """List official TEST image names only; do not open test GT files."""
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing official archive: {ARCHIVE}")
    test = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for m in archive:
            if not m.isfile() or not m.name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            source = "CAR-A" if "/CAR-A/a_test_images/" in m.name else (
                "CAR-B" if "/CAR-B/b_test_images/" in m.name else None
            )
            if source:
                test.append({"source": source, "filename": Path(m.name).name,
                             "archive_path": m.name, "split": "test"})
    test.sort(key=lambda r: (r["source"], r["filename"]))
    if len(test) != EXPECTED_OFFICIAL_TEST:
        raise ValueError(f"Expected {EXPECTED_OFFICIAL_TEST} official test images; got {len(test)}")
    return test


def write_frozen_csv(path, rows, fields):
    """Forbid accidental edits to a previously frozen split manifest."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    new_content = stream.getvalue()
    if path.exists():
        old_content = path.read_text(encoding="utf-8-sig")
        if old_content != new_content:
            raise RuntimeError(f"Existing frozen split differs: {path}. Do NOT overwrite it.")
        return
    path.write_text(new_content, encoding="utf-8")


def show_counts(rows):
    return dict(sorted(Counter((r["source"], int(r["label_length"])) for r in rows).items()))


def main():
    rows = load_training_audit()
    unique, removed = remove_exact_duplicates(rows)
    train, val = stratified_split(unique)
    test = get_test_filename_manifest()

    assert len(train) + len(val) == len(unique)
    assert not {x["sha256"] for x in train} & {x["sha256"] for x in val}
    assert not {(x["source"], x["filename"]) for x in train} & {
        (x["source"], x["filename"]) for x in val
    }
    assert all(x["label"].isdigit() for x in train + val)
    assert not any(x["filename"] == removed[0]["filename"] and x["source"] == removed[0]["source"]
                   for x in train + val)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["source", "filename", "label", "archive_path", "label_length", "sha256",
              "width", "height", "split"]
    write_frozen_csv(SPLITS_DIR / "train.csv", [dict(x, split="train") for x in train], fields)
    write_frozen_csv(SPLITS_DIR / "val.csv", [dict(x, split="val") for x in val], fields)
    write_frozen_csv(SPLITS_DIR / "test_manifest_sealed.csv", test,
                     ["source", "filename", "archive_path", "split"])

    lines = [
        "M1 — FROZEN DATASET SPLITS",
        f"Seed: {SEED}",
        f"Official TRAIN total: {len(rows)}",
        f"Removed exact duplicate from usable manifest: {removed[0]['source']} / {removed[0]['filename']}",
        f"Usable unique images: {len(unique)}",
        f"Train: {len(train)}",
        f"Validation: {len(val)}",
        f"Official TEST file names only (sealed labels): {len(test)}",
        f"Train by CAR source: {dict(Counter(x['source'] for x in train))}",
        f"Validation by CAR source: {dict(Counter(x['source'] for x in val))}",
        f"Train lengths by source/length: {show_counts(train)}",
        f"Validation lengths by source/length: {show_counts(val)}",
        "No exact image SHA256 occurs in both train and validation: TRUE",
        "WRITER DISJOINTNESS: UNKNOWN — writer IDs not available",
        "PERCEPTUAL NEAR DUPLICATES: NOT CHECKED",
        "Original .tar.gz: UNCHANGED",
        "Official test labels: NOT READ / NOT USED FOR SPLIT",
    ]
    summary = AUDIT_OUTPUT_DIR / "split_summary.txt"
    if summary.exists() and summary.read_text(encoding="utf-8") != "\n".join(lines) + "\n":
        raise RuntimeError(f"Existing frozen summary differs: {summary}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print("\nPASS: Frozen train/val manifests created with no exact-duplicate leakage.")
    print("Saved to:", SPLITS_DIR, "and", summary)


if __name__ == "__main__":
    main()

