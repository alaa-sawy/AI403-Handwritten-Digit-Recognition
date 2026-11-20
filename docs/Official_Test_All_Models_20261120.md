# AI403 Official TEST - 21 frozen three-seed evaluations (20 November 2026)

## Scope and verification

- **15 fresh evaluations:** Lab A tested `classic_scratch` and Modern Augmentation ON for seeds 7, 21, 42; Lab B tested `baseline`, `classic_pretrained`, and Modern Augmentation OFF for the same seeds. All 15 completed 6,710/6,710 images, with batch size 32 and frozen checkpoints. Their ZIP exports were checked for privacy, completeness, checkpoint metadata and agreement with per-run JSON records.
- **6 archived evaluations:** Advanced Full and Advanced No-Multiscale, each over the same 6,710-image official benchmark and seeds 7, 21, 42, are separately audited in [M4 Official TEST Evidence](M4_Official_Test_Evidence.md). Their original launch command/Git revision is not independently established.
- The two fresh labs' TEST manifest SHA-256 is identical: `f3a8e41d5c7496aaa61a50e09c99e72705de87d5ca00f419d59a2ef181f607c4`. The saved common evaluator, CTC, registry, preprocessing, dataset-loader, baseline and Modern code hashes also agree across labs.
- The local `src/models/classic.py` SHA-256 differs across A and B. GitHub returns the same stored source blob for this file at the recorded A and B commit IDs, but that does not independently prove byte-identical working files; line endings or local modifications are possibilities. Both evaluations loaded their frozen checkpoint weights strictly.
- Each condition has three independently trained seeds. Tables below report arithmetic mean +/- **sample** SD of percentage metrics (SD in **percentage points**), not a statistical confidence interval. The test set was not used to choose checkpoints.

## Official TEST ranking and metrics

| Condition | CER (%) mean +/- SD (lower) | Exact match (%) mean +/- SD (higher) | Evidence |
| --- | ---: | ---: | --- |
| `baseline` | 78.8814 +/- 0.0966 | 0.0000 +/- 0.0000 | B |
| `classic_scratch` | 2.9328 +/- 0.1792 | 87.7149 +/- 0.6521 | A |
| `classic_pretrained` | 2.3369 +/- 0.2633 | 90.0546 +/- 1.0456 | B |
| `modern` | 4.0271 +/- 0.2613 | 83.3631 +/- 0.9531 | A |
| `modern_off` | 4.0450 +/- 0.2597 | 83.4327 +/- 1.0433 | B |
| `advanced` | 2.4981 +/- 0.3520 | 89.5181 +/- 1.3729 | M4 archive |
| `advanced_no_multiscale` | 2.6055 +/- 0.2495 | 89.1207 +/- 0.9222 | M4 archive |

**Result:** Classic Pretrained has the lowest observed average Official TEST CER (**2.3369%**) and highest average whole-string exact match (**90.0546%**). By contrast, Advanced Full was the lowest-CER *validation* model (2.316%). This is a post-freeze descriptive reversal, not grounds for retuning on TEST.

**Controlled comparisons:** Classic Pretrained beats scratch by 0.5959 percentage points of mean TEST CER, although their validation-selected learning rates also differed. Modern Augmentation ON is 0.0179 pp lower in mean TEST CER than OFF, a small, mixed-seed result. Advanced Full is 0.1074 pp lower than No-Multiscale on mean TEST CER, while their validation means were essentially identical.

## Reproducibility and sources

- [All 21 safe per-seed aggregate rows](../experiments/official_test_final/per_seed_safe.csv) — sample counts, edit totals and frozen-checkpoint SHA-256, **without individual predictions/labels**.
- [All 7 three-seed mean +/- SD summaries](../experiments/official_test_final/summary_safe.csv).
- [Lab A/B provenance and SHA-256](../experiments/official_test_final/provenance_safe.json).
- [Original M4 separately audited results](../experiments/advanced_results/official_test_summary.csv); do not modify M4 source archives or use test results to tune models.
- Raw `best_test_evaluation` directories, `data/splits/test.csv`, TEST photos, ground-truth strings and individual `predictions.csv` are **local only**, not part of GitHub.
- Both fresh labs record PyTorch 2.15.0+cu132. Lab A: Python 3.14.7 / RTX A1000; Lab B: Python 3.13.14 / RTX A2000. M4 data are an earlier archive with a different provenance standard.
- This audit checks structural and arithmetic consistency of the aggregate-only exports; it **cannot rerun every prediction** without the local raw TEST outputs and checkpoints. It should not be misrepresented as a new inference replay.

**Development governance:** Keep the original 501-image validation tables separate from this 6,710-image Official TEST table. Do not re-run training, retune hyperparameters or rank/select new checkpoints using Official TEST.
