# AI403 | Handwritten Digit-String Recognition

**Deep Learning Final Project | Fall 2026 | Pharos University in Alexandria**

**Task:** Recognize an entire variable-length sequence of handwritten digits from a single image of the *courtesy amount recognition* (CAR) field of a bank cheque.

**Approach:** One shared dataset, training protocol, CTC decoder, and evaluation pipeline for a CNN baseline and three deep-learning architecture families (Classic, Modern, Advanced).

> **Project status (20 November 2026):** The shared dataset/training/evaluation pipeline and multiple model families are being integrated by the team. The M4 Advanced model code includes a custom multi-scale CNN block, BiGRU, custom multi-head self-attention, and CTC classifier. The registry supports `advanced`, `advanced_no_multiscale`, and `advanced_no_attention`. Six final Advanced/No-Multiscale checkpoints have passed strict state-dictionary loading and forward-pass checks. A padding audit found that logits and validation metrics can vary with batch composition/size; evaluation batch size must therefore be fixed and reported. The No-Attention variant is implemented in code, but matched three-seed training and evaluation are still pending. See the M4 notes below; do not treat implementation-only ablations as measured findings.

## Quick navigation

- [1. Project goals and hypotheses](#1-project-goals-and-hypotheses)
- [2. Dataset and splits](#2-dataset-and-splits)
- [3. Installation and GPU support](#3-installation-and-gpu-support)
- [4. Quick start: run the baseline](#4-quick-start-run-the-baseline)
- [5. Model interface and architecture plan](#5-model-interface-and-architecture-plan)
- [6. Experiments and fair comparison](#6-experiments-and-fair-comparison)
- [7. Metrics, artifacts, and checkpoints](#7-metrics-artifacts-and-checkpoints)
- [8. Final test protocol](#8-final-test-protocol)
- [9. Repository structure](#9-repository-structure)
- [10. Team workflow](#10-team-workflow)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Credits, citation, and licensing](#12-credits-citation-and-licensing)

## 1. Project goals and hypotheses

**Input:** One cropped image containing a handwritten numeric amount, potentially including background noise or printed cheque marks.

**Output:** One string containing digits `0` to `9`, such as `2250`. The output length is **not fixed**; numbers are treated as strings, never integers.

**Why CTC?** Connectionist Temporal Classification (CTC) supports alignment-free sequence recognition: we have a string label for each image but do not have a bounding box for every digit.

### Planned, testable hypotheses

| ID | Hypothesis | Comparison |
| --- | --- | --- |
| H1 | Adding recurrent sequence modeling improves recognition over the CNN-only baseline. | Classic CNN/ResNet + BiGRU vs. Baseline CNN |
| H2 | A Transformer encoder changes recognition performance compared with a recurrent classic encoder. | Modern CNN+Transformer vs. Classic |
| H3 | Multiscale feature fusion improves recognition over the otherwise matched advanced model without it. | Advanced full vs. Advanced without multiscale |
| H4 | Training-only augmentation improves generalization to unseen images. | Classic Pretrained augmentation ON vs. OFF |

These are **hypotheses, not findings**. Accept or reject each only after recording the controlled experiment results.

## 2. Dataset and splits

We use **[ORAND-CAR-2014](https://www.orand.cl/icfhr2014-hdsr/)** (CAR-A + CAR-B), a public research dataset of handwritten numeric amounts extracted from real bank cheques.

| Partition | CAR-A | CAR-B | Total | Role |
| --- | ---: | ---: | ---: | --- |
| Original official training set | 2,009 | 3,000 | 5,009 | Candidate pool for training/validation |
| Our frozen training split | 1,808 | 2,699 | **4,507** | Model fitting |
| Our frozen validation split | 201 | 300 | **501** | Early stopping, tuning, model selection |
| Original official test set | 3,784 | 2,926 | **6,710** | **Final evaluation only** |

Our usable development pool has **5,008** images after excluding one byte-identical duplicated training image (the original archive remains unchanged). The train/validation split uses seed `403` and stratification by **CAR source and label length**. The frozen manifests are `data/splits/train.csv` and `data/splits/val.csv`.

**Data integrity and limitations:**

- Digit-only labels were checked against the image records; representative samples were visually reviewed.
- No exact image SHA-256 overlap was found across our train/validation split or across the original official training/test partitions.
- **Writer-independent splitting is not proven**, because verified writer IDs were not provided. Perceptual near-duplicate detection was not completed.
- Some cheque images contain punctuation, separators, or currency marks. We follow the dataset's **digits-only ground truth** rather than trying to transcribe these marks.
- The original training set contains digit strings with lengths **2-8**; it does not demonstrate robustness to every kind of handwriting or to leading-zero strings.

### Obtain the dataset

The archive is **not stored in Git**. Each team member can obtain it from:

- **Official download:** [ORAND-CAR-2014.tar.gz](https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz)
- **Team shared drive (restricted access):** `ADD_TEAM_DRIVE_DATASET_LINK_HERE`

The archive is about 194 MiB (203,530,245 bytes). The expected SHA-256 is:

```text
7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a
```

Place the **unchanged** file here:

```text
data/raw/ORAND-CAR-2014.tar.gz
```

On Windows PowerShell, optionally verify the download:

```powershell
Get-FileHash .\data\raw\ORAND-CAR-2014.tar.gz -Algorithm SHA256
```

### Prepare the images

Run the following **from the repository root** after the archive and the frozen `train.csv` / `val.csv` are in place:

```powershell
python data/prepare_images.py
```

This extracts the **5,008** original training/validation image files selected in the manifests into:

```text
data/processed/images/CAR-A/
data/processed/images/CAR-B/
```

> **Important:** Despite the folder name `processed`, these files are original image copies, not pre-resized images. Grayscale conversion, resizing, normalization, and optional augmentation happen in the dataset loader at runtime. This preparation step does **not** open official test labels.

**Common preprocessing (all models):**

1. Convert to grayscale, compositing RGBA images against a white background where needed.
2. Resize to **48 px height**, preserve aspect ratio, keep a **variable image width**.
3. Convert pixels to `float32` and normalize to `[0, 1]`.
4. Pad images to the widest image **within each batch**; keep the original valid widths.
5. For the training split only, optionally use small rotations (about +/-2 degrees) and brightness changes (about +/-10%). Validation and test are never augmented.

**Never regenerate the frozen split for an individual model.** All experiments must use the same committed `train.csv` and `val.csv` files.

## 3. Installation and GPU support

### New team member: clone the repository

After M1 has pushed the project to GitHub, clone it using the **actual repository URL** shared by the team:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_CLONED_REPOSITORY_FOLDER
```

Run all subsequent commands from this root directory. If the repository is private, accept the GitHub collaborator invitation first. Never run another teammate's training job in the same `experiments/` output folder.

### Requirements

- Python compatible with the pinned packages (the training environment was verified with **Python 3.14.7** on Windows 11).
- `requirements.txt` in the repository root.
- An NVIDIA GPU with a compatible driver is recommended for faster training; CPU execution is also supported.
- Enough local disk space for the downloaded archive, extracted images, experiment logs, and checkpoints.

On Windows PowerShell:

```powershell
# Run from the project root.
python --version
python -m pip install -r requirements.txt
```

The repository's single `requirements.txt` pins the **CUDA 13.2** builds of PyTorch and TorchVision on Windows/Linux; on macOS it uses the standard builds. A CUDA build **does not guarantee** that CUDA will be available on every machine: an NVIDIA GPU and suitable driver are required. The current `device: "auto"` trainer chooses **CUDA if available, otherwise CPU**; it does not currently select Apple MPS.

Check PyTorch and GPU detection:

```powershell
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.version.cuda); print('Available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

NVIDIA users can also inspect their driver and VRAM:

```powershell
nvidia-smi
```

**Verified M1 training machine:** Windows 11, Python `3.14.7`, PyTorch `2.15.0+cu132`, NVIDIA RTX A1000 Laptop GPU (**4 GB VRAM**). The CUDA matrix-multiplication test passed. Other team members should record their **own** hardware and library versions; not every computer will deliver the same speed or numerical results.

## 4. Quick start: run the baseline

The commands below are run from the project root (the folder containing `src/`, `configs/`, and `data/`).

### Step 1 - Check the data loader

```powershell
python -m src.datasets.dataset
python -m unittest discover -s tests -p "test_data.py" -v
```

The dataset should report **4,507 training** and **501 validation** images. The data pipeline's four unit tests have passed on the M1 machine.

### Step 2 - Check the shared trainer

```powershell
python -m unittest discover -s tests -p "test_trainer.py" -v
python -m src.experiments_runner --model baseline --stage smoke
```

The shared trainer's six unit tests and baseline smoke test have passed. The smoke test trains on a very small subset; **it is not a model performance result**.

### Step 3 - Tiny-overfit gate (already passed by M1)

```powershell
python -m src.experiments_runner --model baseline --stage tiny-overfit
```

This deliberately checks memorization on four **training** images. M1's test reached `4/4` exact matches after 151 epochs. This does **not** represent validation or official test accuracy. Do not rerun it unnecessarily while another training job is using the GPU.

### Step 4 - Tune the baseline (six predefined trials)

```powershell
python -m src.experiments_runner --model baseline --stage tune
```

The tuning procedure automatically reads `configs/baseline.json` and `configs/protocol.json`, records all six results, then selects the configuration with the **lowest validation character error rate (CER)**.

### Step 5 - Final training (three independent seeds)

Only after tuning has completed:

```powershell
python -m src.experiments_runner --model baseline --stage final
```

This trains **three separate models** using seeds `7`, `21`, and `42`, and reports validation **mean +/- standard deviation**. It does not use official test labels.

### Step 6 - Evaluate an existing checkpoint on validation

```powershell
python -m src.evaluate --checkpoint checkpoints/baseline_results/final/seed_7/best.pt --split val
```

Replace the seed or path with the checkpoint you want to evaluate. The evaluation outputs are saved next to that checkpoint in a `best_val_evaluation/` folder.

## 5. Model interface and architecture plan

All models must implement the **same callable interface**:

```python
logits, output_lengths = model(images, valid_widths)
```

| Tensor | Expected shape | Meaning |
| --- | --- | --- |
| `images` | `[B, 1, 48, Wmax]` | Batch of padded grayscale digit-string images |
| `valid_widths` | `[B]` | Original width of each image before batch padding |
| `logits` | `[B, T, 11]` | Unnormalized class scores at every output time step |
| `output_lengths` | `[B]` | Number of **valid** time steps for each image |

**CTC convention:** blank ID `0`; digit `0` has ID `1`; digit `9` has ID `10`. Preserve repeated and leading digits in the label string (for example, `0052` must stay a string). At training time, CTC receives `log_softmax` scores with shape `[T, B, 11]`.

The encoder must supply valid lengths that reflect any downsampling of image width. CTC feasibility is checked, including extra time steps needed to separate repeated adjacent digits. Models with attention must mask padded sequence positions correctly.

| Registry name | Planned architecture | Owner | Status |
| --- | --- | --- | --- |
| `baseline` | Small CNN + CTC | M1 | Implemented, tests passed; tuning underway |
| `classic_scratch` | ResNet18 (scratch) + BiGRU + CTC | M2 | Awaiting implementation/integration |
| `classic_pretrained` | ResNet18 (pretrained) + BiGRU + CTC | M2 | Awaiting implementation/integration |
| `modern` | CNN + Transformer Encoder + CTC | M3 | Awaiting implementation/integration |
| `advanced` | Custom MultiScale CNN + BiGRU + custom Multi-Head Self-Attention + CTC | M4 | Implemented; six final checkpoints validated; see M4 notes |
| `advanced_no_multiscale` | Same Advanced model without MultiScaleBlock | M4 | Implemented and trained for three seeds; validation/test evidence must be reported with the documented protocol |
| `advanced_no_attention` | Same Advanced model without the attention block | M4 | Registry/model implementation available; matched training and evaluation pending |

**How to add a new model:**

1. Implement the model in `src/models/` with a constructor accepting `dropout`.
2. Ensure `forward(images, valid_widths)` returns `(logits, output_lengths)` as shown above.
3. Use the expected names/classes from `src/models/registry.py` (or update the registry and tests consistently).
4. Add `configs/<model_name>.json` before tuning. The runner currently falls back to `configs/baseline.json` when a model config is missing; **do not rely on that fallback silently** for a new architecture.
5. Pass the unit/forward/CTC/backward, padding-mask (where applicable), smoke, and tiny-overfit checks **before** launching its six tuning trials.

Example (after M2's implementation has been merged and tested):

```powershell
python -m src.experiments_runner --model classic_scratch --stage smoke
python -m src.experiments_runner --model classic_scratch --stage tune
python -m src.experiments_runner --model classic_scratch --stage final
```

## 6. Experiments and fair comparison

**One shared protocol** is used for every architecture: the same frozen split, preprocessing, dataset policy, CTC target encoding, evaluation code, search grid, and selection rule.

### Hyperparameter tuning grid

| Trial | Learning rate | Dropout |
| ---: | ---: | ---: |
| 1 | `0.001` | `0.1` |
| 2 | `0.001` | `0.3` |
| 3 | `0.0003` | `0.1` |
| 4 | `0.0003` | `0.3` |
| 5 | `0.0001` | `0.1` |
| 6 | `0.0001` | `0.3` |

- **Tuning seed:** `403` for all six trials of a given model.
- **Final seeds:** `7`, `21`, `42` (fresh model initialization for each run).
- **Optimizer:** Adam; **loss:** CTC; **augmentation:** enabled during tuning/final training unless a controlled study specifies otherwise.
- **Selection metric:** lowest validation CER; validation CTC loss breaks ties between best epochs. The grid's trial order breaks equal-CER ties when choosing the best trial.
- **Early stopping:** `epochs` is a *maximum*, and `patience` counts consecutive epochs without improvement. The current agreed baseline setup is **batch size 32, maximum 120 epochs, patience 20**; verify the actual committed `configs/baseline.json` before starting the grid.
- **Batch size:** fixed during each six-trial comparison. Document any model-specific adjustments needed to fit GPU memory instead of silently changing the comparison protocol.

The research plan uses **six tuning runs per model version** and **three final seeds per version**. Six trials and 51 total runs are team-planning choices; the course requires a fair tuning study and **at least three seeds per final configuration**, not exactly 51 runs.

### Additional controlled studies

| Study | Comparison | Command (after the prerequisites) |
| --- | --- | --- |
| Augmentation | Classic Pretrained, ON vs. OFF, matched seeds | `python -m src.experiments_runner --model classic_pretrained --stage aug-study` |
| Pretraining | Classic Scratch vs. Classic Pretrained | Tune and finalize both independently, then compare their results |
| Ablation A | Advanced Full vs. No Multiscale, seeds 7/21/42 | Existing results under `experiments/advanced_results/final/` and `ablation_no_multiscale/` |
| Ablation B | Advanced Full vs. No Attention, matched seeds | Implementation exists; training/evaluation has not yet been completed |

**M4 validation summary:** Full Advanced validation CER for seeds 7/21/42: 2.4315%, 2.5473%, 1.9684%. No-Multiscale validation CER: 2.0841%, 2.4701%, 2.3929%. The approximate means are both 2.3157%; the seed-level results vary, so these results do not support a claim that MultiScale improves every run. These are validation results, not a universal performance guarantee.

The No-Attention variant is a code-level ablation at this stage. Do not report an accuracy/CER effect for it until the matched three-seed runs have completed. Run each scientific comparison with aligned splits, preprocessing, tuning/selection policy, and evaluation batch size.

### Planned run budget

| Group | Count |
| --- | ---: |
| Hyperparameter tuning: 5 versions x 6 | 30 |
| Final training: 5 versions x 3 seeds | 15 |
| Advanced no-multiscale ablation: 3 seeds | 3 |
| Classic augmentation OFF: 3 seeds | 3 |
| **Total planned** | **51** |

Do not count **smoke tests**, **tiny-overfit gates**, or **optional troubleshooting runs** as part of these 51 scientific comparisons.

## 7. Metrics, artifacts, and checkpoints

### Metrics

- **Corpus CER** = total Levenshtein edits / total ground-truth characters (lower is better; values can exceed 1 if many extra characters are predicted).
- **Exact-string accuracy** = proportion of entire image labels predicted perfectly (higher is better).
- **Per-digit precision, recall, F1, and substitution/deletion/insertion counts** for digits `0-9`, computed after sequence alignment.
- **Breakdowns** by CAR-A vs. CAR-B and by label length.
- **Across seeds:** mean and sample standard deviation for the final configurations.
- **Computational cost:** training seconds, inference milliseconds per image, parameter count, model/checkpoint size, hardware, CUDA peak *allocated* memory, and observed process RAM (where available).

### M4 checkpoint and padding audit

The six final checkpoints are located at:

```text
checkpoints/advanced_results/final/seed_{7,21,42}/best.pt
checkpoints/advanced_results/ablation_no_multiscale/seed_{7,21,42}/best.pt
```

They were loaded with `torch.load(..., weights_only=True)`, reconstructed from their saved configurations, loaded with `load_state_dict(..., strict=True)`, and passed a forward-shape check producing `[2, 117, 11]`. SHA-256 fingerprints were recorded separately during validation. Checkpoint binaries are not committed to Git; distribute them through the team's restricted Drive link.

The padding audit compared one validation image (width 100) alone and paired with a wider image (width 316), for each of the six checkpoints. The selected image's decoded prediction remained `540`, but valid-position logits differed (maximum absolute differences approximately 0.477–1.281). Full-validation CER/exact-match values also varied across batch sizes 1, 8, 16, 32, and 64. This is evidence of batch-padding sensitivity, not proof that every sample changes prediction. Use one fixed evaluation batch size for reported comparisons and document it. The audit output is `experiments/padding_audit_results.json`.

The six `run_summary.json` files for the final and No-Multiscale runs use project-relative checkpoint paths to improve portability. The original historical tuning/gate summaries are not part of that path rewrite.

### Where results are stored

For the baseline, the runner creates paths such as:

```text
experiments/baseline_results/
  tuning_results.csv
  best_tuned_config.json
  tuning/
    trial_01/
      config.json
      history.csv
      loss_curve.png
      cer_curve.png
      run_summary.json
      validation/
        metrics.json
        per_digit_metrics.csv
        substitution_confusions.csv
        predictions.csv
        errors.csv
    trial_02/ ... trial_06/
  final_runs.csv
  final_mean_sd.json
  final/
    seed_7/
    seed_21/
    seed_42/
```

Model weights are saved separately:

```text
checkpoints/baseline_results/tuning/trial_01/best.pt
checkpoints/baseline_results/final/seed_7/best.pt
```

**GitHub:** commit small `experiments/` logs, configurations, result tables, and plots when they are complete. **Do not commit** raw datasets, extracted images, temporary test labels, large checkpoints, or secrets. Upload final selected model checkpoints to a restricted shared drive and add the download link when available: `ADD_TEAM_DRIVE_CHECKPOINTS_LINK_HERE`.

**Reproducibility:** each run saves its full config and train/validation CSV SHA-256 fingerprints. The runner can reuse a completed tuning run only when its configuration, split hashes, and checkpoint agree. Never overwrite or delete a running trial's output directory.

## 8. Final test protocol

**Test-set governance must be reported accurately.** The project has existing official-test result figures in M4 project records; therefore this README does not claim that the test set is still sealed. Before submission, reconcile the test-evaluation date, exact checkpoint hashes, code version, and whether any model/design decisions were made after seeing test results. Test results must be presented as final evaluation only and must not be used for further tuning or model selection.

For any remaining authorized final evaluation, first confirm the team's frozen protocol and record the exact checkpoint/configuration and evaluation date. The documented preparation/evaluation commands are:

```powershell
python data/prepare_final_test.py --confirm-final
```

Then evaluate the final selected checkpoints, for example:

```powershell
python -m src.evaluate --checkpoint checkpoints/baseline_results/final/seed_7/best.pt --split test --final-test
```

The evaluator writes detailed outputs next to the checkpoint in `best_test_evaluation/`; copy approved **summary tables and plots** into `experiments/` for submission. Keep `data/splits/test.csv`, `data/final_test_images/`, and the original archive **out of Git**. Do not revise the model or tune hyperparameters based on the official test results.

## 9. Repository structure

```text
AI403/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- configs/
|   |-- baseline.json
|   `-- protocol.json
|-- data/
|   |-- raw/                    # local only; NOT committed
|   |-- processed/              # local only; original image cache
|   |-- splits/
|   |   |-- train.csv           # commit frozen manifest
|   |   |-- val.csv             # commit frozen manifest
|   |   `-- test_manifest_sealed.csv  # test names only
|   |-- prepare_images.py
|   |-- make_splits.py
|   `-- prepare_final_test.py
|-- src/
|   |-- datasets/
|   |   |-- preprocess.py
|   |   |-- augment.py
|   |   `-- dataset.py
|   |-- models/
|   |   |-- baseline.py
|   |   `-- registry.py        # other models added by M2-M4
|   |-- ctc_utils.py
|   |-- train.py
|   |-- evaluate.py
|   `-- experiments_runner.py
|-- tests/
|   |-- test_data.py
|   `-- test_trainer.py
|-- experiments/                  # logs, results, plots in Git
|-- checkpoints/                  # local/Drive; NOT committed
|-- demo/                         # to be added by demo owner
|-- report/                       # final report and slides
`-- contributions/                # signed individual logs
```

This is the **target repository organization**; some model files, report files, and demo entry points may still be under integration. The M4 implementation is in `src/models/advanced.py`, with registry selection in `src/models/registry.py` and focused tests in `tests/test_advanced.py`.

Suggested `.gitignore` rules:

```gitignore
data/raw/
data/processed/
data/audit_outputs/
data/final_test_images/
data/splits/test.csv
checkpoints/
__pycache__/
*.pyc
.venv/
.env
*.pt
*.pth
*.tar.gz
*.rar
```

**Do not ignore `experiments/`**: the coursework requires logs, result tables, and plots as evidence. Check `git status` before pushing to prevent accidental dataset or checkpoint uploads.

## 10. Team workflow

| Member | Main implementation responsibility |
| --- | --- |
| **M1** | Dataset selection/audit/splits, data loader, preprocessing/augmentation, CNN+CTC baseline, shared trainer and evaluation infrastructure |
| **M2** | Classic scratch/pretrained architectures and their model-specific checks |
| **M3** | Modern CNN+Transformer architecture and padding-mask checks |
| **M4** | Advanced CNN + BiGRU + custom attention, No-Multiscale study, No-Attention ablation implementation, checkpoint validation, padding audit, and Advanced documentation |
| **M5 / integration owner** | Demo, final integration, comparison/report coordination (confirm assignments with the team) |

Recommended collaboration process:

1. Pull the latest `main` before creating a feature branch (for example `feature/classic`, `feature/modern`, or `feature/advanced`).
2. Keep frozen split manifests and the shared CTC label mapping unchanged.
3. Implement and test each model through the shared interface, and add an explicit model config.
4. Open a pull request; resolve conflicts without overwriting another person's trainer or experiment logs.
5. Merge tested code, run each model's six tuning trials, then its three final seeds.
6. Update `experiments/`, the report tables, and the **individual contribution logs** with commit and run evidence.

Each student's contribution log must describe the actual tasks, hours, evidence, and problems, and be co-signed by another teammate.

**Submission (20 November 2026):** organized repository, dataset documentation, checkpoints or download links, experiment tables and plots, report (15-25 pages), slides, a working live demo, and contribution logs. See the official course brief for exact submission naming and TA instructions.

### Demo status

The live image-to-string demo is a **required deliverable but not yet marked complete** in this README. Once implemented and tested, replace this subsection with the verified command to launch the actual app, plus the path to the final checkpoint and an example input/output. Do **not** invent a demo command before the app exists.

## 11. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `FileNotFoundError: ... data/processed/images/...` | Download the archive to `data/raw/`, confirm frozen `train.csv`/`val.csv` exist, then run `python data/prepare_images.py`. |
| `CUDA available: False` | Check `nvidia-smi`, compatible NVIDIA drivers, the installed PyTorch build, and your environment. The trainer can fall back to CPU. |
| `CUDA out of memory` | Record the hardware and failure; consider reducing per-device batch size or implementing gradient accumulation *with documented effective batch size*. Do not silently alter the experiment protocol mid-grid. |
| CTC output length error | Confirm the CNN width reduction and `output_lengths`, especially for repeated adjacent target digits. |
| Validation CER is high during a smoke test | Smoke/tiny-overfit are correctness gates, **not** measures of generalization. Use full validation after proper training. |
| Missing model implementation | Check the corresponding `src/models/` module, registry entry, config, and model-specific tests; registering a name alone does not implement an architecture. |
| Existing partial run blocks `tune` | Inspect the existing run directory and logs. The runner refuses to reuse an incomplete run to avoid accidental mixing of results. Back it up and use a controlled recovery. |
| A clone lacks the dataset | Raw datasets are intentionally not committed. Use the official source or the restricted team Drive link above. |

## 12. Credits, citation, and licensing

### Dataset citation

Diem, M., Fiel, S., Kleber, F., Sablatnig, R., Saavedra, J. M., Contreras, D., Barrios, J. M., & Oliveira, L. S. (2014). **ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets (HDSRC 2014)**. *International Conference on Frontiers in Handwriting Recognition (ICFHR)*.

Official project and dataset: <https://www.orand.cl/icfhr2014-hdsr/>

**Dataset license:** [Creative Commons BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Credit the dataset creators; use it only in accordance with the license, and **do not publish modified dataset images** in this repository. The data remains separate from the project's code and any code license the team may later choose.

**Frameworks/tools:** PyTorch, TorchVision (where used by the Classic models), Pillow, NumPy, Matplotlib, and psutil. The M4 attention operation is implemented as a custom PyTorch module rather than calling `nn.MultiheadAttention` in the model forward path. Cite any third-party architecture implementations or pretrained weights actually used in the final report.

### Team and links (fill before submission)

- **Team name:** `ADD_TEAM_NAME`
- **Members / student IDs:** `ADD_MEMBER_NAMES_AND_IDS`
- **Repository:** `ADD_GITHUB_REPOSITORY_LINK`
- **Dataset Drive:** `ADD_TEAM_DRIVE_DATASET_LINK_HERE`
- **Final checkpoints Drive:** `ADD_TEAM_DRIVE_CHECKPOINTS_LINK_HERE`
- **Report / slides / demo:** link the finished artifacts here after verification.

---

*This README documents the implementation and experiment protocol. Experimental results are reported only when a run has completed and the saved metrics have been verified. The original dataset and course requirements remain the authoritative sources for licensing and submission rules.*
