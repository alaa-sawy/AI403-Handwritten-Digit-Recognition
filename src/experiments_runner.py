"""Fair experiment runner: six LR/dropout trials, three seeds, data study, ablation.

Commands (from AI403 project root):
    python -m src.experiments_runner --model baseline --stage smoke
    python -m src.experiments_runner --model baseline --stage tune
    python -m src.experiments_runner --model baseline --stage final
    python -m src.experiments_runner --model baseline --stage aug-study

Run the stages separately after checking estimated epoch time and resources.
"""

import argparse
import csv
import json
import statistics
from copy import deepcopy
from pathlib import Path

from src.train import ROOT, run_experiment, split_fingerprint


def results_dir(model):
    folder = "baseline_results" if model == "baseline" else model + "_results"
    return ROOT / "experiments" / folder


def load_protocol():
    return json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))


def load_config(model):
    path = ROOT / "configs" / f"{model}.json"
    if not path.exists():
        # Uniform training defaults; save a dedicated config as new models arrive.
        path = ROOT / "configs" / "baseline.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["model"] = model
    return config


def write_csv(path, rows):
    if not rows:
        raise ValueError("No runs to summarize")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        keys = list(rows[0])
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_or_reuse(config, out_dir):
    """Resume long tuning/final studies without repeating completed runs."""
    summary_path = out_dir / "run_summary.json"
    if summary_path.is_file():
        old_config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        if old_config != config:
            raise ValueError(f"Existing run has DIFFERENT configuration: {out_dir}")
        old_result = json.loads(summary_path.read_text(encoding="utf-8"))
        if old_result.get("split_sha256") != split_fingerprint():
            raise ValueError(f"Frozen train/val manifests changed since this run: {out_dir}")
        if not Path(old_result["checkpoint_path"]).exists():
            raise FileNotFoundError(f"Missing checkpoint for completed run: {out_dir}")
        print(f"Reusing completed run: {out_dir}")
        return old_result
    if out_dir.exists():
        raise RuntimeError(f"Incomplete run at {out_dir}. Inspect it and move aside before retrying.")
    return run_experiment(config, out_dir)


def simple_row(result):
    return {key: result[key] for key in (
        "model", "seed", "learning_rate", "dropout", "augmentation",
        "best_epoch", "val_cer", "val_exact_match", "train_seconds",
        "inference_ms_per_image", "parameter_count", "checkpoint_size_mb")}


def tune(model):
    protocol = load_protocol()
    base = load_config(model)
    out = results_dir(model)
    results = []
    number = 0
    for learning_rate in protocol["learning_rates"]:
        for dropout in protocol["dropouts"]:
            number += 1
            config = deepcopy(base)
            config.update(learning_rate=learning_rate, dropout=dropout,
                          seed=protocol["tuning_seed"],
                          augmentation=protocol["tuning_augmentation"])
            print(f"\n{model} TRIAL {number}/6: lr={learning_rate} dropout={dropout}", flush=True)
            result = run_or_reuse(config, out / "tuning" / f"trial_{number:02d}")
            results.append({"trial": number, **simple_row(result)})
            write_csv(out / "tuning_results.csv", results)
    # Validate all trials on the SAME frozen validation set.
    best = min(results, key=lambda row: (row["val_cer"], row["trial"]))
    best_config = deepcopy(base)
    best_config.update(learning_rate=best["learning_rate"], dropout=best["dropout"],
                       augmentation=protocol["tuning_augmentation"])
    (out / "best_tuned_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\nBEST trial {best['trial']}: CER={best['val_cer']:.4f}; "
          f"LR={best['learning_rate']}; dropout={best['dropout']}")


def tuned_config(model):
    path = results_dir(model) / "best_tuned_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Tuning not finished. Missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(rows, group_name):
    """Mean and sample SD: each seed trains independently."""
    summary = {"group": group_name, "seeds": [r["seed"] for r in rows]}
    for key in ("val_cer", "val_exact_match", "train_seconds", "inference_ms_per_image"):
        data = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = statistics.mean(data)
        summary[f"{key}_sd"] = statistics.stdev(data) if len(data) >= 2 else None
    return summary


def final(model):
    protocol = load_protocol()
    base = tuned_config(model)
    out = results_dir(model)
    results = []
    for seed in protocol["final_seeds"]:
        config = deepcopy(base)
        config["seed"] = seed
        print(f"\nFINAL {model}, seed={seed}", flush=True)
        result = run_or_reuse(config, out / "final" / f"seed_{seed}")
        results.append(simple_row(result))
        write_csv(out / "final_runs.csv", results)
    summary = aggregate(results, model)
    (out / "final_mean_sd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFINAL mean CER +/- SD:", summary["val_cer_mean"], summary["val_cer_sd"])


def augment_study(model):
    """Paired controlled study: reuse final ON seeds; train only three OFF seeds."""
    protocol = load_protocol()
    base = tuned_config(model)
    if not base["augmentation"]:
        raise ValueError("Study requires the final tuned configuration with augmentation ON")
    out = results_dir(model)
    paired = []
    for seed in protocol["final_seeds"]:
        on_path = out / "final" / f"seed_{seed}" / "run_summary.json"
        if not on_path.exists():
            raise FileNotFoundError(f"Run FINAL stage first: {on_path}")
        on_result = json.loads(on_path.read_text(encoding="utf-8"))
        config = deepcopy(base)
        config.update(seed=seed, augmentation=False)
        print(f"\nAUGMENTATION OFF: {model}, seed={seed}", flush=True)
        off_result = run_or_reuse(config, out / "augmentation_off" / f"seed_{seed}")
        paired.extend([{"condition": "ON", **simple_row(on_result)},
                       {"condition": "OFF", **simple_row(off_result)}])
        write_csv(out / "augmentation_study.csv", paired)
    for condition in ("ON", "OFF"):
        vals = [r for r in paired if r["condition"] == condition]
        stats = aggregate(vals, f"{model}_augmentation_{condition}")
        print(f"{condition}: mean CER {stats['val_cer_mean']:.4f} "
              f"+/- {stats['val_cer_sd']:.4f}")


def ablation():
    """Advanced full vs no-multiscale: same seeds and selected hyperparameters."""
    protocol = load_protocol()
    base = tuned_config("advanced")
    out = results_dir("advanced")
    runs = []
    for seed in protocol["final_seeds"]:
        full_path = out / "final" / f"seed_{seed}" / "run_summary.json"
        if not full_path.exists():
            raise FileNotFoundError(f"First finish advanced FINAL seed {seed}")
        full = json.loads(full_path.read_text(encoding="utf-8"))
        config = deepcopy(base)
        config.update(seed=seed, model="advanced_no_multiscale")
        no_multi = run_or_reuse(config, out / "ablation_no_multiscale" / f"seed_{seed}")
        runs.extend([{"condition": "FULL", **simple_row(full)},
                     {"condition": "NO_MULTISCALE", **simple_row(no_multi)}])
        write_csv(out / "ablation_results.csv", runs)
    for condition in ("FULL", "NO_MULTISCALE"):
        vals = [r for r in runs if r["condition"] == condition]
        stats = aggregate(vals, condition)
        print(f"{condition}: mean CER {stats['val_cer_mean']:.4f} "
              f"+/- {stats['val_cer_sd']:.4f}")


def tiny_overfit(model):
    """Deliberately memorize four training examples; not a generalization result."""
    config = load_config(model)
    config.update(augmentation=False, dropout=0.0, learning_rate=0.005,
                  epochs=300, patience=300, batch_size=4, seed=403, tiny_overfit_gate=True)
    out = results_dir(model) / "gates" / "tiny_overfit"
    result = run_experiment(config, out, tiny_overfit=True)
    if result["val_exact_match"] == 1.0:
        print("PASS: all four training strings were memorized.")
    else:
        raise AssertionError("Tiny-overfit did NOT reach 4/4 yet. Inspect learning before tuning.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline",
                        choices=["baseline", "classic_scratch", "classic_pretrained", "modern", "advanced"])
    parser.add_argument("--stage", required=True,
                        choices=["smoke", "tiny-overfit", "tune", "final", "aug-study", "ablation"])
    args = parser.parse_args()
    if args.stage == "smoke":
        config = load_config(args.model)
        run_experiment(config, smoke=True)
    elif args.stage == "tiny-overfit":
        tiny_overfit(args.model)
    elif args.stage == "tune":
        tune(args.model)
    elif args.stage == "final":
        final(args.model)
    elif args.stage == "aug-study":
        augment_study(args.model)
    elif args.stage == "ablation":
        if args.model != "advanced":
            parser.error("Ablation is defined for the advanced model only")
        ablation()


if __name__ == "__main__":
    main()
