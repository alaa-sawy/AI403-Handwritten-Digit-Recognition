# M4 Official TEST: archived evidence and audit

**Project work window (team-reported): 17-20 November 2026.**  
**Archived test-result file modification dates:** 18 November 2026, according to ZIP member metadata; these timestamps **do not independently prove the original execution date/time**.

## Data sources and verification scope

The project owner supplied `checkpoints-20261120T170122Z-1-001.zip` and `M4_Official_Test_Evidence_Package.zip`. The former had SHA-256 `dc8d28a0ad457f83c690713d9d428727bf4ee288a9a387c9075524541cc5893e`. Six saved `best_test_evaluation` folders were audited. Each contained **6,710** Official TEST predictions (**3,784 CAR-A + 2,926 CAR-B**) and **33,506** reference characters. Character error rate (CER), exact-string match, edit totals, per-digit metrics and confusion records were independently recalculated from the saved CSVs. **Zero mismatches** were found in these checks.

This audit validates the **consistency of saved files**. It is **not** a fresh inference run and does **not** certify the full original execution environment or the timing of model selection.

## Test results by seed

| Model | Seed | Official TEST CER (%) | Exact match (%) | Samples | Total edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| advanced | 7 | 2.6831 | 88.7630 | 6,710 | 899 |
| advanced | 21 | 2.7189 | 88.6885 | 6,710 | 911 |
| advanced | 42 | 2.0922 | 91.1028 | 6,710 | 701 |
| advanced_no_multiscale | 7 | 2.4085 | 89.9404 | 6,710 | 807 |
| advanced_no_multiscale | 21 | 2.5219 | 89.2996 | 6,710 | 845 |
| advanced_no_multiscale | 42 | 2.8861 | 88.1222 | 6,710 | 967 |

### Three-seed summary (mean +/- sample SD; percentages / percentage points)

| Condition | Official TEST CER (%) | Exact match (%) |
| --- | ---: | ---: |
| Advanced Full | 2.4981 +/- 0.3520 | 89.5181 +/- 1.3729 |
| Advanced No-Multiscale | 2.6055 +/- 0.2495 | 89.1207 +/- 0.9222 |

The Full condition shows a **0.1074 percentage-point** lower mean TEST CER, but the paired seed results vary, and the prior Validation means were practically identical. **Do not treat the difference as proof of a consistent MultiScale benefit; do not use TEST scores to change model selection or tune weights.**

## Checkpoint identity

All six archived `best.pt` checkpoints passed archive-level metadata/strict-load checks against M4's saved training-run metadata: model identifier, seed, best epoch, batch size **32**, recorded validation CER and frozen split hashes. The checkpoint SHA-256 values from the independent audit are:

| Model | Seed | Best epoch | Checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| advanced | 7 | 26 | `b9af175e6ead901d29901adeea51bd77fa822ef7ded60372c1a73781854a38b1` |
| advanced | 21 | 24 | `ce6afa01e43fcfc0ff0a2d34cbbb70d2969d43704a4fa813fe8452d377c0ab29` |
| advanced | 42 | 51 | `49e3933aeeb9df70b08e90210af0d2756a69126aca972172d82d685c7a856a4d` |
| advanced_no_multiscale | 7 | 75 | `da0a5d8ba6b1c6f96b9e4c14040a8c098e02637a634e2c18dea6e56f3d2d2b05` |
| advanced_no_multiscale | 21 | 76 | `e21534c2111c768966de4790039c0eae7b67fd508680fef52f9819f0a6484689` |
| advanced_no_multiscale | 42 | 43 | `9cd4a2374d1a21ebbe2cbf489ef8b035e7c18ad0a8200e8c37af7fe107ab8572` |

The source ZIP used `checkpoints/advanced_no_multiscale_results/ablation_no_multiscale/` for the No-Multiscale condition; this repository's canonical local organization is `checkpoints/advanced_results/ablation_no_multiscale/`. The checkpoint bytes/hashes, not the folder spelling, identify each artifact.

## Original-run provenance: known vs unknown

- **Project activity period:** 17-20 November 2026 (reported by the project owner; not a claim that each test ran on all four dates).
- **Archive file modification evidence:** the six saved `metrics.json` files are dated **18 November 2026** in the supplied ZIP. Modification times may be preserved or changed during copying/archiving; do not treat them as signed run logs.
- **Batch size:** 32 in the corresponding checkpoint configurations; the current shared `src/evaluate.py` reads the batch size from the checkpoint. The original launch command was **not preserved**.
- **Original evaluation Git commit and software environment:** **not separately recorded** in the supplied files. Current GitHub source/commits cannot prove the historical checkout used for the test.
- **Whether any design or checkpoint decisions followed inspection of TEST:** **not independently documented**. This should be disclosed and no further model selection should be performed on TEST.
- **Original training result references:** `experiments/advanced_results/final/`, `experiments/advanced_results/ablation_no_multiscale/`, and `experiments/padding_audit_results.json`.

## Repository privacy and hand-in handling

This repository stores **aggregate TEST metrics and SHA-256 fingerprints only**. The original `predictions.csv` and `errors.csv` include Official TEST ground-truth labels; keep them in restricted Drive or on the evaluation machine, **not GitHub**. Do not commit `data/splits/test.csv` or extracted TEST images. The results above concern **M4 only**; they are not the final consolidated M1-M4 Official TEST comparison.
