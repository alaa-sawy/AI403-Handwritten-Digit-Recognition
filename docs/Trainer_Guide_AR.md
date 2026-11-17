# AI403 - Shared Trainer (M1 + M5 handoff)

## Goal

ONE training protocol for ALL 5 model configurations, NOT a temporary baseline trainer.
Keep the already-tested `src/datasets/{dataset,preprocess,augment}.py` unchanged.
Only replace `src/models/baseline.py` (adds real configurable dropout).

## Put the files into your existing project

```
AI403/
  configs/
    baseline.json                NEW
    protocol.json                NEW
  data/
    prepare_final_test.py        NEW (DO NOT run until FINAL evaluation)
  src/
    ctc_utils.py                 NEW
    train.py                     NEW - shared trainer for every model
    evaluate.py                  NEW - shared metrics/evaluation for every model
    experiments_runner.py        NEW - tuning/seeds/augmentation/ablation
    models/
      registry.py                NEW - model names and constructors
      baseline.py                REPLACE - same CNN + configurable dropout
  tests/
    test_trainer.py              NEW
  experiments/                   important training runs only
  checkpoints/                   optional copies of chosen best.pt checkpoints
```

### Dependencies

```powershell
python -m pip install torch pillow numpy psutil matplotlib
```

## Execute safely, in this order

### Gate A - fast unit tests

```powershell
python -m unittest discover -s tests -p "test_trainer.py" -v
```

### Gate B - one real mini-training epoch (8 TRAIN images + 4 VAL images)

```powershell
python -m src.experiments_runner --model baseline --stage smoke
```

No checkpoint or final result is written in smoke stage.

### Gate C - tiny overfit 4 TRAIN images

```powershell
python -m src.experiments_runner --model baseline --stage tiny-overfit
```

This deliberately checks if the model can memorize four inputs. **Do not launch a 6-trial grid if this gate fails.**

### Tuning (6 runs per model: 3 learning rates x 2 dropouts)

```powershell
python -m src.experiments_runner --model baseline --stage tune
```

All six trials share the same split, preprocessing, optimizer (Adam), batch size, epoch budget, augmentation policy, and tuning seed (403). The only two tuned hyperparameters are learning rate and dropout.
Results: `experiments/baseline_results/tuning_results.csv` and `best_tuned_config.json`.

### Final model (3 independent restarts)

```powershell
python -m src.experiments_runner --model baseline --stage final
```

Seeds 7, 21, 42. Best checkpoint chosen on validation CER only.
Results in `experiments/baseline_results/final_runs.csv`, `final_mean_sd.json`, plus individual checkpoints, logs, and detailed digit metrics.

### Controlled augmentation study

```powershell
python -m src.experiments_runner --model baseline --stage aug-study
```

Must run `final` before `aug-study`. Reuses the final 3 ON runs, trains 3 new OFF runs with exactly the same architecture, LR, dropout, batch, split and seeds. The shared Playbook proposes the **Classic** model as primary data-study, so after M2 integrates the classic models, run the corresponding command with `--model classic_pretrained` too if that version is selected.

### Adding other architectures later

Add each model class with constructor `dropout=...` and `forward(images, valid_widths) -> (logits, output_lengths)`:

- `classic_scratch` & `classic_pretrained`: `src/models/classic.py`, class `ClassicCTC(dropout, pretrained)`
- `modern`: `src/models/modern.py`, class `ModernCTC(dropout)`
- `advanced` & `advanced_no_multiscale`: `src/models/advanced.py`, class `AdvancedCTC(dropout, multiscale)`

The names above are already in `src/models/registry.py`. Do NOT create dummy models to fill the gap: wait for M2-M4's actual tested code. Use the same training API for every new model.

For each model:

```powershell
python -m src.experiments_runner --model classic_scratch --stage smoke
python -m src.experiments_runner --model classic_scratch --stage tiny-overfit
python -m src.experiments_runner --model classic_scratch --stage tune
python -m src.experiments_runner --model classic_scratch --stage final
```

Ablation (after full Advanced final):

```powershell
python -m src.experiments_runner --model advanced --stage ablation
```

Optional override for direct one-off troubleshooting (NOT a tuning-grid result):

```powershell
python -m src.train --smoke
python -m src.train --model baseline --lr 0.0003 --dropout 0.1 --epochs 2 --run-name debug_baseline
```

### Model-level checks BEFORE a real 6-trial sweep

- Mixed-width batch returns correct `widths`
- Input `[B,1,48,W]` and output `[B,T,11]`, output lengths `[B]`
- CTC has enough time steps for repeated digits (`T >= len(label) + adjacent repeats`)
- Finite CTC loss and gradients, parameter update
- Attention padding masks ignore invalid sequence positions (Modern/Advanced owners must unit-test this)
- Tiny overfit (4 real images)

## Training outputs / rubric trace

Inside each `experiments/<model>_results/<stage>/...` run folder (checkpoints are stored separately):

- `config.json`: full hyperparameter configuration for reproduction
- `history.csv`, `loss_curve.png`, `cer_curve.png`: train loss, validation loss, CER, exact-match, time per epoch
- `checkpoints/<model>_results/<stage>/.../best.pt`: checkpoint and its saved configuration + split checksums
- `run_summary.json`: best epoch, CER/exact, parameters, checkpoint size, training/inference time, GPU memory, hardware, split hashes
- `validation/metrics.json`: corpus CER, exact-match, length and CAR-A/CAR-B breakdown
- `validation/per_digit_metrics.csv`: 0..9 precision/recall/F1 + substitutions/deletions/insertions
- `validation/substitution_confusions.csv`: true-digit -> predicted-digit confusion counts
- `validation/errors.csv`, `predictions.csv`: real examples for the report

The *final model families* have mean and sample SD over the three independent final seeds; full report tables, literature/hypotheses, dataset licensing, individual contributions and demo are separate team deliverables, not created automatically by the trainer.

## Official test policy

Do **NOT** run these until tuning, final model choices, ablations and rules are FROZEN. The official test is not allowed to affect model selection.

```powershell
python data/prepare_final_test.py --confirm-final
python -m src.evaluate --checkpoint checkpoints/baseline_results/final/seed_7/best.pt --split test --final-test
```

Evaluate the selected final seeds/models exactly once at the end, and report their test mean/SD; don't revise settings using test feedback.

Add the following to `.gitignore`:

```gitignore
data/raw/
data/processed/
data/audit_outputs/
data/final_test_images/
data/splits/test.csv
checkpoints/
```

The repository should store frozen `train.csv`/`val.csv`, config files and experiment summaries, **not** raw data or final test labels. If model weights are too large, host them in the shared drive and put the link in README.

## Notes

- 6 trials per model and 51 runs are OUR planning choices, not the professor's required exact counts.
- Five model versions: baseline, classic scratch, classic pretrained, modern, advanced.
- 51 = 30 tuning + 15 final + 3 advanced ablation + 3 classic augmentation-OFF. A *separate* baseline augmentation study adds 3 extra OFF runs (54 total).
- 3 final seeds per configuration are mandatory. Compute costs before you commit to all runs.
- CPU `psutil` RSS is an observed process memory estimate; GPU memory is the peak allocated PyTorch CUDA memory. Neither measures the full machine RAM.
- Checkpoints and PyTorch experiments are only reproducible within the documented environment/device; floating-point results can vary across hardware or library versions.
