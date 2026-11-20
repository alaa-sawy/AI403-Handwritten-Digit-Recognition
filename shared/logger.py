"""M5 CSV logger for reproducible experiment events."""
import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["timestamp_utc", "model", "stage", "seed", "epoch", "train_loss",
          "val_loss", "val_cer", "val_exact_match", "elapsed_seconds"]

class TrainingLogger:
    def __init__(self, path="logs/m5_training_events.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, **values):
        row = {key: values.get(key, "") for key in FIELDS}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
