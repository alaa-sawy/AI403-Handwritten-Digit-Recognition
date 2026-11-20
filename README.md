# AI403 - Handwritten Digit-String Recognition

**Deep Learning Final Project | Pharos University in Alexandria | Fall 2026**

**Git repository:** [AI403-Handwritten-Digit-Recognition](https://github.com/alaa-sawy/AI403-Handwritten-Digit-Recognition)  
**Team checkpoint Drive (shared by the project owner):** [Google Drive folder](https://drive.google.com/drive/folders/1y6Tq6SnGyZ_oa1Vexl4Ovsx27PQArUZL?usp=sharing)

> **Project status — 20 November 2026:** All five primary model versions have recorded six-trial tuning and three-seed **validation** results. A three-seed Modern augmentation ON/OFF study and Advanced no-multiscale ablation are complete. **This is not yet the full course submission:** a reconciled official TEST comparison, integrated demo, final team report/slides and all individual contribution logs are not verified in `main`. No new training is required merely to read the saved results.

## Navigation

1. [Task and hypotheses](#1-task-and-hypotheses)
2. [Dataset and preprocessing](#2-dataset-and-preprocessing)
3. [Model implementations](#3-model-implementations)
4. [Final validation comparison](#4-final-validation-comparison)
5. [Training and tuning protocol](#5-training-and-tuning-protocol)
6. [Controlled studies: pretraining, augmentation, ablation](#6-controlled-studies-pretraining-augmentation-ablation)
7. [Error analysis and computational cost](#7-error-analysis-and-computational-cost)
8. [Setup and evaluation commands](#8-setup-and-evaluation-commands)
9. [Official TEST governance](#9-official-test-governance)
10. [Checkpoint storage and reproducibility](#10-checkpoint-storage-and-reproducibility)
11. [Repository contents and tests](#11-repository-contents-and-tests)
12. [Deliverables still to finalize](#12-deliverables-still-to-finalize)
13. [Team responsibilities and references](#13-team-responsibilities-and-references)

## 1. Task and hypotheses

**Input:** one image of a handwritten numeric string in the courtesy-amount-recognition (CAR) region of a bank cheque. **Output:** the complete ordered digit string, not an integer or a single digit. Leading zeros and repeated digits must be preserved (e.g. `00520`).

All primary models implement end-to-end sequence transcription with **Connectionist Temporal Classification (CTC)** rather than explicitly segmenting each character. CTC blank is ID `0` and digits `0`–`9` use class IDs `1`–`10`.

| ID | Research hypothesis | Evidence from current validation |
| --- | --- | --- |
| H1 | Contextual sequence modeling improves over a small CNN baseline. | Both Classic models greatly outperform the Baseline. Other architecture components also change, so this is not proof of recurrence as the sole cause. |
| H2 | A Transformer is a useful alternative to recurrent sequence modeling. | Modern works, but has worse validation CER than Classic; it has lower measured model-forward latency on the recorded (not identical) hardware. |
| H3 | MultiScale improves Advanced recognition. | **Not supported** by the matched three-seed mean: Full and No-Multiscale CER are essentially equal. |
| H4 | Training-only augmentation improves Modern generalization. | Modern Augmentation ON has slightly lower mean validation CER than OFF, with mixed seed-level effects. |

Interpretations above are limited to the **501-image validation split**, not an independent official TEST conclusion.

## 2. Dataset and preprocessing

**Source:** [ORAND-CAR-2014](https://www.orand.cl/icfhr2014-hdsr/), using CAR-A and CAR-B. Original download: [ORAND-CAR-2014.tar.gz](https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz).

| Partition | CAR-A | CAR-B | Total | Use |
| --- | ---: | ---: | ---: | --- |
| Original TRAIN | 2,009 | 3,000 | 5,009 | Development source |
| Frozen project TRAIN | 1,808 | 2,699 | **4,507** | Training |
| Frozen project VALIDATION | 201 | 300 | **501** | Early stopping, tuning, error analysis |
| Official TEST | 3,784 | 2,926 | **6,710** | Controlled final evaluation |

One exact duplicate training image (`b_car_004864.png`) was removed from the usable development manifest, leaving **5,008** samples. Split seed **403** was stratified by CAR subset and string length. All reported model runs use the same committed `data/splits/train.csv` and `data/splits/val.csv` fingerprints.

**Audit scope:** no exact SHA-256 image overlap was detected between train/validation or original TRAIN/TEST; **writer independence and absence of perceptual near-duplicates were not proven**. Test labels were not required for the split construction. See [dataset documentation](data/DATASET.md).

Shared preprocessing: transparent pixels composited on white; grayscale; aspect-ratio-preserving resize to **48-pixel height**; `float32` values in `[0,1]`; right-padding within each batch with per-image valid widths. Training-only augmentation (when enabled) rotates within **+/-2 degrees** and varies brightness by about **+/-10%**. Neither validation nor TEST is augmented.

The original archive is **not committed to Git**. Put it at `data/raw/ORAND-CAR-2014.tar.gz`; the recorded SHA-256 is:

```text
7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a
```

## 3. Model implementations

All models implement `logits, output_lengths = model(images, valid_widths)`, where images have shape `[B,1,48,Wmax]` and CTC logits have shape `[B,T,11]`. Models with width-downsampling report appropriately shorter valid time lengths.

| Registry model name | Architecture | Parameters | Saved checkpoint MiB |
| --- | --- | ---: | ---: |
| `baseline` | CNN + CTC | 94,091 | 0.36 |
| `classic_scratch` | ResNet18 (scratch) + BiGRU + CTC | 13,541,323 | 51.73 |
| `classic_pretrained` | ResNet18 (ImageNet pretrained) + BiGRU + CTC | 13,541,323 | 51.73 |
| `modern` | CNN + Transformer + CTC | 1,183,179 | 5.51 |
| `advanced` | MultiScale CNN + BiGRU + custom Attention + CTC | 1,245,867 | 4.77 |

- **M1 Baseline:** small CNN, per-timestep linear classifier and CTC, without explicit GRU/Transformer.
- **M2 Classic:** ResNet18 adapted for grayscale and horizontal sequence preservation, two-layer bidirectional GRU, CTC. Scratch and pretrained versions use the **same architecture**, but different backbone initialization. The pretrained variant starts from TorchVision's ImageNet weights.
- **M3 Modern:** CNN, sinusoidal positions, two Transformer encoder layers with four attention heads and padding masking, CTC.
- **M4 Advanced:** custom 3x3/5x5 MultiScale CNN feature block, BiGRU, custom four-head self-attention, residual feed-forward, CTC.
- **M4 Ablation:** `advanced_no_multiscale` removes the custom MultiScale block. `advanced_no_attention` is implemented in code, but matched three-seed results **have not been recorded**.

Sources: [model registry](src/models/registry.py) and [implementations](src/models/).

## 4. Final validation comparison

**These are VALIDATION metrics, NOT official TEST scores.** Each primary model was trained independently using final seeds **7, 21 and 42**. Reported numbers are **mean +/- sample SD**; SD for percentage metrics is in **percentage points**.

| Model | CER (%) - lower better | Exact-string match (%) - higher better | Tuned LR | Dropout |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 78.361 +/- 0.118 | 0.000 +/- 0.000 | 0.001 | 0.1 |
| Classic Scratch | 3.242 +/- 0.253 | 85.828 +/- 0.998 | 0.0003 | 0.1 |
| Classic Pretrained | 2.676 +/- 0.161 | 88.090 +/- 0.756 | 0.0001 | 0.1 |
| Modern | 4.052 +/- 0.241 | 83.367 +/- 1.170 | 0.001 | 0.1 |
| Advanced | 2.316 +/- 0.306 | 89.621 +/- 1.584 | 0.0003 | 0.1 |

**Lowest mean validation CER among the five primary models:** Advanced, **2.316%**, with mean exact-string match **89.621%**. The Baseline has **78.361%** CER and **0%** exact-string matches; it is a deliberately simple but weak reference, not a working high-accuracy recognizer.

All individual-seed results, losses, predictions, error analyses and curves are saved under:

- [M1 Baseline](experiments/baseline_results/)
- [M2 Classic Scratch](experiments/classic_scratch_results/)
- [M2 Classic Pretrained](experiments/classic_pretrained_results/)
- [M3 Modern](experiments/modern_results/)
- [M4 Advanced](experiments/advanced_results/)

## 5. Training and tuning protocol

The controlled search uses exactly the same grid **for each of the five primary model versions**:

| Trial | LR | Dropout |
| ---: | ---: | ---: |
| 01 | 0.001 | 0.1 |
| 02 | 0.001 | 0.3 |
| 03 | 0.0003 | 0.1 |
| 04 | 0.0003 | 0.3 |
| 05 | 0.0001 | 0.1 |
| 06 | 0.0001 | 0.3 |

- **Tuning seed:** `403`, using validation CER to choose the best trial.
- **Final seeds:** `7, 21, 42`, independently initialized and fitted using the chosen per-model configuration.
- **Training:** Adam, batch size 32, CTC loss, weight decay 0.0001, gradient clip 5.0, maximum 120 epochs, early stopping patience 20.
- **Augmentation:** ON in regular tuning/final runs; OFF only for the controlled Modern comparison.
- **Recorded study counts:** 30 tuning + 15 primary final + 3 Modern augmentation OFF + 3 Advanced no-multiscale = **51 scientific runs**. Internal smoke/tiny-overfit gates are not included.

**Chosen tuning configurations (seed 403 is not a fourth final seed):**

| Model | Best trial | LR | Dropout | Best tuning val CER (%) |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1 | 0.001 | 0.1 | 78.271 |
| `classic_scratch` | 3 | 0.0003 | 0.1 | 2.316 |
| `classic_pretrained` | 5 | 0.0001 | 0.1 | 2.740 |
| `modern` | 1 | 0.001 | 0.1 | 3.744 |
| `advanced` | 3 | 0.0003 | 0.1 | 1.775 |

The committed files `experiments/<model>_results/best_tuned_config.json` and `final/seed_<N>/config.json` are authoritative for historical runs. Base `configs/*.json` can contain pre-tuning defaults.

## 6. Controlled studies: pretraining, augmentation, ablation

### Pretrained vs scratch — M2

Under the same data split, model structure and tuning budget, **Classic Scratch** records **3.242%** mean final-validation CER versus **2.676%** for **Classic Pretrained**. Pretrained is lower by **0.566 percentage points**. Each was independently tuned and therefore uses different selected learning rates; this is not an experiment isolating only the initial weights while keeping all optimization settings fixed.

### Augmentation ON vs OFF — M3 Modern (actual executed study)

Matched seeds and final learning rate `0.001` / dropout `0.1`; OFF changes only the augmentation switch, while ON reuses final Modern runs.

| Condition | Validation CER mean (%) | Sample SD (pp) |
| --- | ---: | ---: |
| ON | 4.052 | 0.241 |
| OFF | 4.194 | 0.226 |

The ON mean is lower by **0.142 pp**, but the per-seed differences are not uniformly positive. Evidence: [`experiments/modern_results/augmentation_study.csv`](experiments/modern_results/augmentation_study.csv). **The older claim that this study was run for Classic Pretrained was incorrect.**

### Ablation — M4 Full Advanced vs No-Multiscale

Matched seeds `7,21,42`, with selected LR `0.0003` / dropout `0.1` and shared preprocessing.

| Condition | Validation CER mean (%) | Sample SD (pp) |
| --- | ---: | ---: |
| Advanced Full | 2.316 | 0.306 |
| Advanced No-Multiscale | 2.316 | 0.204 |

Means are essentially **identical at reported precision**. Seed effects vary in direction, so there is **no demonstrated mean validation advantage** from MultiScale. No-Attention is **implementation only**. Evidence: [`ablation_results.csv`](experiments/advanced_results/ablation_results.csv).

## 7. Error analysis and computational cost

**Metrics:** corpus character error rate (total edit distance / total ground-truth characters), whole-string exact match, per-digit precision/recall/F1 from sequence alignment, and substitutions/deletions/insertions; broken down by CAR source and string length.

**Recorded validation failures (seed 7; examples are pairs from committed `errors.csv` files):**

| Model | Filename | Ground truth | Prediction | Observed error |
| --- | --- | --- | --- | --- |
| Baseline | `a_car_000207.png` | `10000` | `10` | Many deleted digits |
| Classic Scratch | `a_car_000212.png` | `15366` | `15866` | Substitution |
| Classic Pretrained | `a_car_000451.png` | `23104` | `218` | Several edits |
| Modern | `a_car_000384.png` | `2510` | `1510` | Substitution |
| Advanced | `a_car_000384.png` | `2510` | `250` | Deletion |

The Baseline heavily collapses to predicting `10`; its saved diagnostics report extensive deletions and zero exact matches across three seeds. The cause is **not proven** to be solely lack of recurrence or a fault in CTC. See [`Baseline_Analysis.md`](Baseline_Analysis.md) and the per-seed `validation/` error files. Raw dataset image redistribution must respect the documented licence.

**Computational cost:** mean +/- sample SD across final seeds; inference timing below is **model-forward only** (not image loading, preprocessing, decoding or full app latency).

| Model | Train minutes mean +/- SD | Forward ms/image mean +/- SD | Seed 7 peak GPU allocated MiB | Seed 7 GPU |
| --- | ---: | ---: | ---: | --- |
| Baseline | 27.1 +/- 14.2 | 0.694 +/- 0.477 | 615.8 | RTX A1000 Laptop |
| Classic Scratch | 63.8 +/- 14.6 | 3.503 +/- 0.185 | 1987.8 | RTX A1000 Laptop |
| Classic Pretrained | 100.9 +/- 34.5 | 3.040 +/- 0.127 | 1988.0 | RTX A2000 Laptop |
| Modern | 46.5 +/- 38.4 | 0.365 +/- 0.064 | 429.6 | RTX A1000 Laptop |
| Advanced | 30.2 +/- 7.4 | 1.869 +/- 0.209 | 1465.4 | RTX 3060 Laptop |

Different machines/GPU types and process interruptions limit direct wall-time comparisons. GPU values measure **PyTorch-allocated** memory, not total hardware memory. The [Advanced padding audit](experiments/padding_audit_results.json) also shows batch-size-dependent logits/validation metrics; use a documented **fixed evaluation batch size** (the historical evaluation used batch size 32) and disclose this sensitivity in comparisons.

## 8. Setup and evaluation commands

Clone the project and use a **consistent Python interpreter**. Windows PowerShell example:

```powershell
git clone https://github.com/alaa-sawy/AI403-Handwritten-Digit-Recognition.git
cd AI403-Handwritten-Digit-Recognition
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
```

The committed requirements pin Torch `2.15.0` / TorchVision `0.29.1` (CUDA 13.2 builds on Windows/Linux). Use a Python version supported by those package wheels; team runs used Windows 11, Python 3.13/3.14 and different NVIDIA GPUs. CPU fallback is available. A virtual environment that imports Python but cannot import `torch` is **not** ready for model evaluation.

Download the unchanged ORAND archive and verify the SHA above, then run:

```powershell
& .\.venv\Scripts\python.exe data/prepare_images.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Validation evaluation** after placing the selected checkpoint from Drive at the canonical local path:

```powershell
& .\.venv\Scripts\python.exe -m src.evaluate --checkpoint checkpoints/advanced_results/final/seed_7/best.pt --split val
```

The evaluator loads the **model weights and stored config** from the checkpoint and writes `best_val_evaluation/` beside it. This command does **not** train the model.

**Reproduction stages** (run only in a prepared, isolated experiment workspace; not needed to read existing results):

```powershell
python -m src.experiments_runner --model baseline --stage tune
python -m src.experiments_runner --model classic_scratch --stage final
python -m src.experiments_runner --model classic_pretrained --stage final
python -m src.experiments_runner --model modern --stage aug-study
python -m src.experiments_runner --model advanced --stage ablation
```

The paired `aug-study` and `ablation` stages assume their corresponding `final` runs have already been completed. **Do not rerun completed training just to build the report.**

## 9. Official TEST governance

The saved model results above use **VALIDATION, not TEST**. There are **no verified consolidated cross-model official TEST result tables** in the reviewed GitHub `main`. Existing M4 documentation references some historical official-test figures, so **do not falsely claim that official TEST has never been opened**.

Before any new TEST operation, reconcile previously accessed test data/results, the evaluation date, checkpoint SHA-256, code revision, selected models and batch size. **Freeze all choices before final evaluation** and never select/tune using test scores.

The project supports `data/prepare_final_test.py --confirm-final` to create the locally stored 6,710-sample TEST manifest/images and `python -m src.evaluate --checkpoint <checkpoint> --split test --final-test` for an authorized final pass. The preparation script will refuse to overwrite `data/splits/test.csv`. Avoid repeated TEST exposure when earlier results might already exist. Any approved test summary must be clearly labelled and recorded separately from validation results; never commit test labels or extracted images.

## 10. Checkpoint storage and reproducibility

The team-provided checkpoint Drive folder is [here](https://drive.google.com/drive/folders/1y6Tq6SnGyZ_oa1Vexl4Ovsx27PQArUZL?usp=sharing). **Its current contents were not independently enumerated in this README audit**. The project owner reports that model weights are being organized there. All **21** checkpoint locations required for the recorded full/final studies are:

```text
checkpoints/
  baseline_results/final/seed_{7,21,42}/best.pt
  classic_scratch_results/final/seed_{7,21,42}/best.pt
  classic_pretrained_results/final/seed_{7,21,42}/best.pt
  modern_results/final/seed_{7,21,42}/best.pt
  modern_results/augmentation_off/seed_{7,21,42}/best.pt
  advanced_results/final/seed_{7,21,42}/best.pt
  advanced_results/ablation_no_multiscale/seed_{7,21,42}/best.pt
```

A team Drive upload may use alternative folder names (notably `advanced_no_multiscale_results/ablation_no_multiscale`). Map downloaded files to the **canonical local paths** above to run the shared evaluator, without changing their contents. Interrupted runs and `gates/` checkpoint folders are not final deliverables. Large checkpoints belong in Drive; small logs/JSON/CSV/plots belong in Git.

**Historical exception:** three small Baseline `best.pt` files are still tracked on GitHub, despite new `checkpoints/` uploads being excluded by `.gitignore`. Removing these should be a *separate* clean-up only after their Drive copies are verified.

`experiments/` holds saved tuning/final records, loss/CER plots, per-digit tables, predictions, error analyses, split SHA-256s, hardware and run summaries. **Gate output directories are deliberately not versioned**, though test code remains.

**Reproducibility caution:** several historical `run_summary.json` entries contain **absolute checkpoint paths from the original laptops**. `src/experiments_runner.py` validates those paths before reusing prior output directories. A fresh clone with committed logs but missing historical checkpoints can refuse `tune`/`final` reuse; restoring the correct checkpoint and adjusting to a controlled isolated reproduction workflow is safer than overwriting old results. Do not mix checkpoints from different seeds or models.

## 11. Repository contents and tests

Verified main-branch source structure:

```text
AI403-Handwritten-Digit-Recognition/
  README.md                 requirements.txt
  configs/                  data/                 src/
  src/models/{baseline,classic,modern,advanced,registry}.py
  src/{ctc_utils,train,evaluate,experiments_runner,modern_gate}.py
  tests/{test_data,test_trainer,test_classic,test_modern,test_advanced}.py
  experiments/              docs/
  Baseline_Analysis.md      M1_Contribution_Log.md
  AI403_M4_Advanced_Architecture_and_Ablation_Report_M4_Validation_Added.docx
```

`demo/`, an integrated `report/` with official final PDF/slides, and complete `contributions/` are **not present in the reviewed main snapshot**. The M4 DOCX is *model-specific supporting documentation*, not the final team report.

**Tests (no new training):**

```powershell
python -m unittest discover -s tests -v
```

Smoke and tiny-overfit checks are local debugging tools and do **not** replace full three-seed validation or official TEST. No Gate result/checkpoint package is required for Drive submission.

## 12. Deliverables still to finalize

The course brief requires a **15–25 page report with seven mandatory tables**, a **15-minute presentation**, a **5-minute live demo**, complete experiment/error analyses, reproducible checkpoints and individual co-signed contribution logs.

| Deliverable | Current verified status |
| --- | --- |
| Five primary models + three-seed validation | **Recorded on GitHub** |
| Modern augmentation / Advanced No-Multiscale | **Recorded on GitHub** |
| Checkpoint package | **User reports Drive uploads**; final 21-file audit recommended |
| Official TEST | **Reconcile prior M4 evidence and publish a verified final comparison** |
| Live demo | **Under development by M5**; no confirmed app/launch command on `main` |
| Team report PDF, slides | Not yet present as unified final artifacts on `main` |
| Individual contribution + peer evaluation | M1 log present; other signed records not verified |
| Team name, IDs, final report/demo links | Need verified team details |

**Official hand-in:** one ZIP named `DL_Project_TeamName.zip` containing `TeamName_Report.pdf`, `TeamName_Slides.pdf` (or PPTX) and a text file with the Git repository URL. Add the supervising TA as repository collaborator. Put large data/checkpoints in Drive and include the relevant links here. The course brief lists **20 November 2026** as the deadline. Do not upload raw dataset images to GitHub.

## 13. Team responsibilities and references

**Team name, members and student IDs:** not yet verified; add them before official submission rather than inventing identities.

| Workstream | Recorded responsibility |
| --- | --- |
| M1 | Dataset audit/splits, shared pipeline, CNN-CTC baseline |
| M2 | ResNet18 scratch/pretrained Classic, tuning and final comparisons |
| M3 | CNN-Transformer, augmentation ON/OFF and masking checks |
| M4 | Custom MultiScale/attention Advanced, no-multiscale ablation, padding audit |
| M5 | Integration, demo and final hand-in coordination (in progress) |

**References and external components**

1. Diem et al. (2014), *ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets.* [Official dataset](https://www.orand.cl/icfhr2014-hdsr/).
2. Graves et al. (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.* ICML.
3. Shi, Bai and Yao (2017), *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition.* [arXiv](https://arxiv.org/abs/1507.05717).
4. He et al. (2016), *Deep Residual Learning for Image Recognition.* [arXiv](https://arxiv.org/abs/1512.03385).
5. Vaswani et al. (2017), *Attention Is All You Need.* [arXiv](https://arxiv.org/abs/1706.03762).
6. [PyTorch](https://pytorch.org/), [TorchVision ResNet18/ImageNet weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), Pillow, NumPy, Matplotlib and psutil.

The source documentation records the dataset as **CC BY-NC-ND 4.0**; comply with attribution, noncommercial use and distribution restrictions. External pretrained components must be cited and adapted. Prediction code does not rely on third-party AI inference APIs.

---

*All numeric performance claims in this README are derived from committed validation CSV/JSON records. Only verified final TEST results may subsequently be labelled official TEST.*
