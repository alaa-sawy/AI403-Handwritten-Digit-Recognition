# M5 — Shared System, Evaluation, Computational Cost and Demo

## Scope and implementation

M5 consumes the integrated M1–M4 project rather than reimplementing model architectures. The repository already provides one shared CTC training path (`src/train.py`), a model registry (`src/models/registry.py`), common preprocessing/data loading (`src/datasets/`), a common evaluation implementation (`src/evaluate.py`), experiment configurations (`configs/`), and per-run history/configuration/summary artifacts (`experiments/`).

M5 adds:
- A checkpoint-backed Streamlit demo in `demo/app.py`, using the same preprocessing and CTC greedy decoder as the training/evaluation pipeline.
- A checkpoint loader in `src/inference.py` that reconstructs the architecture and strictly loads the selected trained weights. Classic-pretrained inference does not redownload ImageNet weights because the trained checkpoint already contains the complete model state.
- Shared metric/evaluator and computational-cost wrappers in `shared/`.
- A reproducible validation comparison CSV and three figures in `m5/results/`.
- This report and M5 integration tests.

## Validation comparison (three final seeds)

All values below are derived from the committed `final_runs.csv` files, or from the three matched `NO_MULTISCALE` rows in `experiments/advanced_results/ablation_results.csv`. CER and exact match are mean ± sample standard deviation across seeds. They are validation results, not official-test results.

| Model | Validation CER (%) ↓ | Exact match (%) ↑ | Train time (min) | Inference (ms/image) | Parameters | Checkpoint (MB) |
| baseline | 78.361 ± 0.118 | 0.00 ± 0.00 | 27.1 | 0.694 | 94,091 | 0.36 |
| classic_scratch | 3.242 ± 0.253 | 85.83 ± 1.00 | 63.8 | 3.503 | 13,541,323 | 51.73 |
| classic_pretrained | 2.676 ± 0.161 | 88.09 ± 0.76 | 100.9 | 3.040 | 13,541,323 | 51.73 |
| modern | 4.052 ± 0.241 | 83.37 ± 1.17 | 46.5 | 0.365 | 1,183,179 | 5.51 |
| advanced | 2.316 ± 0.306 | 89.62 ± 1.58 | 30.2 | 1.869 | 1,245,867 | 4.77 |
| advanced_no_multiscale | 2.316 ± 0.204 | 89.16 ± 1.00 | 41.8 | 1.665 | 1,163,691 | 4.46 |\n
## Computational-cost interpretation and limitations

Parameter counts, checkpoint sizes, and recorded inference/training times are read from the final experiment artifacts. The original runs were performed on more than one GPU model and software environment (the run summaries identify RTX A1000 Laptop, RTX A2000 Laptop, and RTX 3060 hardware across model families). Therefore, recorded training and inference time are descriptive historical measurements and should not be treated as a strict controlled speed ranking across architectures. The demo's inference time depends on the user's current device.

The Baseline is retained as the required pipeline reference, but its recorded validation CER is high and its exact-match score is zero. It is not a reliable user-facing recognizer. The best validation CER among the listed final model families is Advanced. No-Multiscale is reported separately as the M4 ablation, not as a sixth independently tuned model family.

## Official-test evidence supplied by M4

M4 supplied an official-test summary for Advanced and Advanced-No-Multiscale:
- Advanced: test CER 2.4981% ± 0.3520; exact match 89.5181% ± 1.3729.
- Advanced-No-Multiscale: test CER 2.6055% ± 0.2495; exact match 89.1207% ± 0.9222.

These figures are transcribed from `M4_Official_Test_Evidence_Package/official_test_summary.csv`. The accompanying provenance document still contains unfilled fields for evaluation timestamp, commit, command/version, batch size, and post-test decision history. M5 therefore preserves the supplied numbers but does not claim complete provenance or independently re-run the official test.

## Test and demo commands

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m streamlit run demo/app.py
```

The full test suite includes model contract/unit tests and M5 checkpoint integration checks. The pretrained Classic tests may need the torchvision ResNet18 ImageNet weights cached or internet access; the M5 demo itself does not require an ImageNet download when loading the trained checkpoint.

## Dataset and test-set governance

The raw ORAND-CAR-2014 archive and extracted images are local-only and excluded from Git. The fixed train/validation manifests remain unchanged. The official test labels are not included in this M5 package. M4's supplied official-test summary is reported as existing evidence, with its provenance limitation explicitly noted.

## M4 handoff limitation relevant to integration

The supplied M4 contribution log states that the `advanced_no_attention` variant is implemented and registered, but matched three-seed training/evaluation has not been completed. There is therefore no final no-attention checkpoint family to include in the demo or cross-model results table. The M5 demo includes the trained Advanced and Advanced-No-Multiscale variants that have complete supplied checkpoint sets.

