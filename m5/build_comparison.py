"""Rebuild M5 validation comparison tables and figures from saved run CSVs."""
import csv, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "m5" / "results"
ARCHS = ["baseline", "classic_scratch", "classic_pretrained", "modern", "advanced"]

def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    rows = []
    for arch in ARCHS:
        runs = read_rows(ROOT / "experiments" / f"{arch}_results" / "final_runs.csv")
        values = {key: [float(r[key]) for r in runs] for key in
                  ("val_cer", "val_exact_match", "train_seconds",
                   "inference_ms_per_image", "parameter_count", "checkpoint_size_mb")}
        rows.append({
            "model": arch, "seeds": len(runs),
            "val_cer_mean_pct": 100 * statistics.mean(values["val_cer"]),
            "val_cer_sd_pct_points": 100 * statistics.stdev(values["val_cer"]),
            "exact_match_mean_pct": 100 * statistics.mean(values["val_exact_match"]),
            "exact_match_sd_pct_points": 100 * statistics.stdev(values["val_exact_match"]),
            "train_time_mean_min": statistics.mean(values["train_seconds"]) / 60,
            "inference_ms_per_image_mean": statistics.mean(values["inference_ms_per_image"]),
            "parameters": int(values["parameter_count"][0]),
            "checkpoint_size_mb": statistics.mean(values["checkpoint_size_mb"]),
        })
    ablation = read_rows(ROOT / "experiments" / "advanced_results" / "ablation_results.csv")
    no_multi = [r for r in ablation if r["condition"] == "NO_MULTISCALE"]
    rows.append({
        "model": "advanced_no_multiscale", "seeds": len(no_multi),
        "val_cer_mean_pct": 100 * statistics.mean(float(r["val_cer"]) for r in no_multi),
        "val_cer_sd_pct_points": 100 * statistics.stdev(float(r["val_cer"]) for r in no_multi),
        "exact_match_mean_pct": 100 * statistics.mean(float(r["val_exact_match"]) for r in no_multi),
        "exact_match_sd_pct_points": 100 * statistics.stdev(float(r["val_exact_match"]) for r in no_multi),
        "train_time_mean_min": statistics.mean(float(r["train_seconds"]) for r in no_multi) / 60,
        "inference_ms_per_image_mean": statistics.mean(float(r["inference_ms_per_image"]) for r in no_multi),
        "parameters": int(float(no_multi[0]["parameter_count"])),
        "checkpoint_size_mb": statistics.mean(float(r["checkpoint_size_mb"]) for r in no_multi),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "model_comparison_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [r["model"].replace("_", "\n") for r in rows]
    charts = [
        ("val_cer_mean_pct", "Validation CER by model", "CER (%)", "validation_cer_comparison.png"),
        ("exact_match_mean_pct", "Validation exact-match accuracy", "Exact match (%)", "validation_exact_match_comparison.png"),
        ("inference_ms_per_image_mean", "Recorded inference time", "Milliseconds / image", "inference_time_comparison.png"),
    ]
    for key, title, ylabel, filename in charts:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, [r[key] for r in rows])
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)
    print(f"Saved comparison CSV and 3 charts under {OUT}")

if __name__ == "__main__":
    main()
