# M3 — Modern CNN–Transformer–CTC: Implementation and Validation Results

**Project:** AI403 — Handwritten Digit-String Recognition\
**Dataset:** ORAND-CAR-2014 (CAR-A and CAR-B)\
**Status:** Modern ON and OFF validation completed and committed; these historical results are validation only. Official TEST for both conditions is processed separately and is not yet represented by an audited repository-wide M3 TEST summary.\
**Evidence:** `src/models/modern.py`, `src/modern_gate.py`, `tests/test_modern.py`, `configs/modern.json`, and `experiments/modern_results/`.

## 1. Scope and hypothesis

M3 implements the **modern** architecture family for variable-length handwritten-digit-string transcription. It supports the project hypothesis that a Transformer-based sequence encoder may perform differently from the recurrent classic model. That hypothesis cannot be accepted or rejected from M3 results alone; it requires the agreed final cross-model comparison.

The system predicts a **digit string** rather than a number, preserving order and repetitions. The dataset is split using the shared frozen manifests: **4,507 training** and **501 validation** samples. No official test images or labels were used for the experiments reported here.

## 2. Architecture

**Input:** `[B, 1, 48, W]`, grayscale images of height 48 and variable width. The shared collator pads images on the right and supplies their valid widths.

| Component | Implementation |
|---|---|
| CNN feature extractor | Three 3×3 convolutional blocks, with 32, 64, and 128 output channels, BatchNorm and ReLU |
| Spatial reduction | Two 2×2 MaxPool layers reduce time width by 4×; adaptive average pooling reduces feature height to 1 |
| Feature projection | 128 → 256 |
| Position information | Sinusoidal positional encoding |
| Sequence model | Two Transformer encoder layers; 4 attention heads; `d_model=256`; feed-forward dimension 512 |
| Padding handling | Invalid input/feature positions masked; Transformer key-padding mask based on post-CNN valid lengths |
| Classifier | Linear 256 → 11; CTC blank ID 0 plus ten digit classes |
| Parameters | **1,183,179** |

The model returns logits `[B, T, 11]` and valid output lengths, conforming to the shared trainer/evaluator interface. The architecture uses the team's own CNN/sequence assembly, masking logic and positional encoding while relying on documented PyTorch Transformer primitives.

## 3. Training and evaluation protocol

All experiments used the project's shared preprocessing and frozen train/validation split, Adam optimizer, CTC loss, validation-CER model selection, early stopping and the same tuning grid used for the other models.

| Setting | Value |
|---|---|
| Tuning learning rates | 0.001, 0.0003, 0.0001 |
| Tuning dropout rates | 0.1, 0.3 |
| Tuning seed | 403 |
| Final seeds | 7, 21, 42 |
| Batch size | 32 |
| Maximum epochs | 120 |
| Early-stopping patience | 20 |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Gradient clipping | 5.0 |
| Training augmentation | Enabled; validation augmentation disabled |
| Loss and primary selection metric | CTC loss; validation character error rate (CER) |

### Hyperparameter study — all six trials

| Trial | Learning rate | Dropout | Best epoch | Validation CER ↓ | Exact match ↑ |
|---:|---:|---:|---:|---:|---:|
| **01** | **0.001** | **0.1** | **51** | **3.7437%** | **84.03%** |
| 02 | 0.001 | 0.3 | 60 | 4.2455% | 81.04% |
| 03 | 0.0003 | 0.1 | 115 | 4.1683% | 82.44% |
| 04 | 0.0003 | 0.3 | 101 | 3.9367% | 83.23% |
| 05 | 0.0001 | 0.1 | 87 | 5.6735% | 76.85% |
| 06 | 0.0001 | 0.3 | 93 | 5.6735% | 76.05% |

**Selected configuration:** Trial 01, learning rate **0.001**, dropout **0.1**, chosen by the minimum validation CER. It was then independently retrained with each final seed. Trial selection did **not** use official test scores.

## 4. Final independent-seed validation

| Seed | Best epoch | Validation CER ↓ | Exact match ↑ | Train time (minutes) | Inference (ms/image) |
|---:|---:|---:|---:|---:|---:|
| 7 | 94 | 3.8595% | 84.23% | 89.91 | 0.439 |
| 21 | 85 | 3.9753% | 83.83% | 32.34 | 0.326 |
| 42 | 61 | 4.3227% | 82.04% | 17.22 | 0.331 |
| **Mean ± sample SD** | — | **4.0525% ± 0.2410 percentage points** | **83.3666% ± 1.1696 percentage points** | **46.49 ± 38.35** | **0.365 ± 0.064** |

All three seeds were evaluated on the same **501-image validation set**. These are **validation metrics, not official test metrics**. The standard deviations describe seed-to-seed variation, not confidence intervals for unseen test data.

## 5. Per-digit and error analysis (final seed 7, validation only)

Seed 7 produced **422 correct entire-string predictions and 79 incorrect strings** across 501 validation images, with **CER 3.8595%**.

The 100 character-level edit operations were:

| Error type | Count | Share of edits |
|---|---:|---:|
| Substitutions | 72 | 72% |
| Deletions | 14 | 14% |
| Insertions | 14 | 14% |
| **Total** | **100** | **100%** |

The least accurate digits by per-digit F1 were **9 (0.9383; support 125)**, **2 (0.9415; support 201)** and **8 (0.9450; support 149)**; digit **0 (0.9872; support 940)** was strongest on this validation subset. These support counts are unequal, so a raw comparison of F1 requires that caveat.

The most frequent observed substitution directions were **3→2 (7)**, **6→0 (5)**, **9→8 (4)**, **6→8 (3)** and **4→1 (3)**. The following are concrete examples extracted from `errors.csv`:

| Filename | Ground truth | Prediction | Edit count | Observable error |
|---|---|---|---:|---|
| `a_car_000212.png` | `15366` | `15866` | 1 | Digit substitution (3→8) |
| `a_car_000384.png` | `2510` | `1510` | 1 | Digit substitution (2→1) |
| `b_car_003252.png` | `243718` | `24378` | 1 | Digit deletion (1) |
| `b_car_003138.png` | `330000` | `20000` | 2 | Wrong leading digits and shortened sequence |
| `b_car_003478.png` | `369626` | `30026` | 3 | Multiple errors in one string |

**Interpretation:** Substitution errors dominate the observed edit counts. Image-dependent causes such as similar handwriting shapes, touching strokes, spacing or cheque background should be treated as **hypotheses**, not established explanations solely from the CSV. Selected example images can be shown and discussed in the final team report, subject to dataset licensing and attribution. **Do not commit raw extracted dataset images or the temporary error-sample ZIP.**

Full per-digit precision/recall/F1 and all errors are available under each final seed's `validation/` directory, including `per_digit_metrics.csv`, `errors.csv`, `predictions.csv` and `substitution_confusions.csv`.

## 6. Computational cost

| Metric | Recorded value |
|---|---|
| GPU | NVIDIA RTX A1000 Laptop GPU |
| Model parameters | 1,183,179 |
| Checkpoint size | ~5.51 MiB per final seed |
| Peak PyTorch GPU memory allocation | ~429.5 MiB per final seed |
| Mean training time | 46.49 minutes (large observed variance) |
| Mean model-forward inference | 0.365 ms per image across final seeds |

**Measurement caveats:** The GPU-memory figure is **PyTorch peak allocated memory**, not total VRAM occupancy. Inference timings measure **model forward computation**, not loading, preprocessing or end-to-end application latency. Wall-clock train times differ substantially across seeds and may include host interruptions, so they should not be treated as isolated architectural-performance measurements.

## 7. Verification and reproducibility

Verified on the M3 laptop (`D:\AI403`):

- The `tests/test_modern.py` suite completed with **5/5 passing** tests.
- All **six tuning and three final checkpoints** loaded successfully with `strict=True`; their model configurations, train/validation split fingerprints and seeds matched expectations.
- Every final seed has **501 predictions**, **10 per-digit metric rows**, and all expected validation-evaluation files.
- The M3 tiny-overfit gate memorized **4/4 training strings**, reaching CER **0** and exact match **1.0** after **51 epochs**; this is a debugging gate, not a generalization score.
- Best-model checkpoints and their training summaries are stored locally. Supply either final checkpoints through the team shared drive or complete reproduction instructions, consistent with the final submission protocol.

Useful commands from the repository root:

```powershell
python -m unittest discover -s tests -p "test_modern.py" -v
python -m src.modern_gate
python -m src.experiments_runner --model modern --stage smoke
python -m src.experiments_runner --model modern --stage tune
python -m src.experiments_runner --model modern --stage final
```

**Do not rerun completed training unnecessarily.** The main experiment runner reuses completed runs under its recorded-configuration and split-fingerprint checks.

## 8. Integration and hand-off

The Modern source, six tuning trials, three final ON seeds, and three matched augmentation OFF runs are committed to the team repository. The official TEST is used only after checkpoint selection has been frozen, and its outputs must be documented separately from all validation numbers above.

For the project-wide demo, the shared inference adapter and checkpoint paths must be verified and the interface launched with a saved trained checkpoint. No repeated training is required merely to integrate these stored results.

*All numeric model-performance figures in this document are derived from saved M3 validation records, not Official TEST metrics.*
