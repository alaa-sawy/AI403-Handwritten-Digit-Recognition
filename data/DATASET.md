# Dataset Documentation — ORAND-CAR-2014

**Project:** AI403 — Handwritten Digit-String Recognition  
**Dataset component owner:** M1  
**Dataset release:** ORAND-CAR-2014, released for the ICFHR 2014 Handwritten Digit String Recognition competition  
**Status:** Frozen train/validation split; Official TEST is for frozen-model final evaluation only. Archived M4 test aggregates have been documented; the other families' test summaries are not yet consolidated on GitHub.

## 1. Dataset identity, source, and license

The project uses **ORAND-CAR-2014**, specifically **CAR-A** and **CAR-B**, which contain digit-string images cropped from the courtesy-amount-recognition (CAR) fields of real bank cheques. The two subsets come from different sources and contain real-world handwriting variation, background patterns, cheque marks, and other noise. The model transcribes the **digit string**, not a numeric amount or its currency symbols.

- Official dataset and competition page: https://www.orand.cl/icfhr2014-hdsr/
- Official archive: https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz
- Original publication: Diem, M., Fiel, S., Kleber, F., Sablatnig, R., Saavedra, J. M., Contreras, D., Barrios, J. M., & Oliveira, L. S. (2014). *ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets (HDSRC 2014).* International Conference on Frontiers in Handwriting Recognition (ICFHR).
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**), as stated by the official publisher: https://creativecommons.org/licenses/by-nc-nd/4.0/

This repository distributes scripts and metadata manifests, **not the original images/archive** or transformed image files. Users must follow the original dataset's license and attribution requirements. The archive is used for noncommercial academic research.

## 2. Original dataset and project split

| Partition | CAR-A | CAR-B | Total | Project use |
| --- | ---: | ---: | ---: | --- |
| Original official TRAIN | 2,009 | 3,000 | **5,009** | Candidate pool for project train/validation |
| Frozen project TRAIN | 1,808 | 2,699 | **4,507** | Model fitting |
| Frozen project VALIDATION | 201 | 300 | **501** | Model selection, early stopping, tuning, error analysis |
| Original official TEST | 3,784 | 2,926 | **6,710** | Final evaluation **only**, after all decisions are frozen |

A single **byte-identical training duplicate** (`CAR-B/b_car_004864.png`) was excluded from the project's usable development manifest. The original archive was **not changed**. Therefore, the development pool contains **5,008** unique training-origin images, divided into 4,507 train and 501 validation examples. The full original dataset contains 11,719 images, before this internal duplicate exclusion.

The train/validation split was generated **only from the official TRAIN partition**, using:

- Fixed random seed **403**.
- Approximately 90% train / 10% validation (exact counts above).
- Stratification by **CAR subset (A or B)** and **ground-truth string length**.
- Preservation of existing rare strata in training where necessary.
- Permanent committed manifests at `data/splits/train.csv` and `data/splits/val.csv`; every model uses these **same files**.
- A separate `data/splits/test_manifest_sealed.csv` containing official TEST **filenames only**, not labels.

Dataset labels are strings of ASCII digits, not integer values. This preserves leading zeros and repeated adjacent digits. Original official TRAIN label lengths observed in the audit range from **2 to 8 digits**. No model may use the official TEST labels for tuning, selecting epochs, building hypotheses, or error inspection during development.

## 3. Data audit and leakage controls

The audit decoded **5,009 original training images**, finding **zero unreadable images**, **zero missing training labels**, **one exact duplicate-image group**, and **zero conflicting labels among identical-image duplicates**. Its training-image counts were CAR-A = 2,009 and CAR-B = 3,000.

**Verified controls:**

1. No exact SHA-256 image-content hash appears in both frozen train and validation manifests.
2. The audit found **zero exact image-hash matches** between the original TRAIN and TEST image partitions. The TEST images were hashed for this integrity check **without opening TEST ground-truth labels**.
3. The original TEST labels were not accessed for the split or training/validation experiment selection.
4. The frozen split writer rejects attempts to overwrite existing manifests with different content.

**Limits — do not overclaim:** Writer-disjoint splitting **cannot be verified**, because no verified writer IDs were available to the project. **Perceptual near-duplicate search was not completed**; different image bytes might still contain visually near-identical material. Hence the evidence establishes **no exact-hash overlap**, not an absolute guarantee of no possible leakage. A representative sample-review manifest exists, but a completed, auditable set of manual review annotations is not established by the committed records.

Audit evidence: `data/audit_outputs/image_audit_summary.txt`, `data/audit_outputs/leakage_check_summary.txt`, `data/audit_outputs/split_summary.txt`, `data/audit_outputs/data_audit_images.csv`, and `data/audit_outputs/duplicate_groups.csv`.

## 4. Shared preprocessing and augmentation

**All models** use the same data loader and preprocessing protocol:

1. Read image and convert it to grayscale (`L`); if alpha is present, composite onto a **white background** first.
2. Resize to **48 pixels in height**, preserving aspect ratio and variable width (Lanczos resampling).
3. Convert to `float32` and normalize pixel values to **[0, 1]**.
4. Supply a tensor `[1, 48, width]` per image; pad batch images to `[B, 1, 48, max_width]` with white (`1.0`) pixels, and retain each image's valid original width for sequence-length handling and CTC.
5. Apply augmentation **only to training images**, if the experiment config enables it: random rotation in **[-2°, +2°]** (expanded white canvas) and brightness multiplier **[0.90, 1.10]**. Never augment validation or test data.

`data/processed/images/` is a local cache of the **original selected image bytes**, not a cache of pre-resized or already augmented images. Actual grayscale conversion, resizing, normalization, and optional augmentation happen in `src/datasets/dataset.py` at loading time using `src/datasets/preprocess.py` and `src/datasets/augment.py`.

The target CTC vocabulary contains **11 classes**: blank ID `0` plus digit IDs `1` through `10` representing the characters `0` through `9`, respectively.

## 5. Reproducibility and access

1. Obtain the original archive from the official URL above; place the unchanged file at `data/raw/ORAND-CAR-2014.tar.gz`. The raw archive is excluded from Git.
2. The original archive fingerprint recorded in the project README is SHA-256:
   `7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a`
3. From the project root, verify the downloaded archive (Windows PowerShell):

   ```powershell
   Get-FileHash .\data\raw\ORAND-CAR-2014.tar.gz -Algorithm SHA256
   ```

4. Use the **already committed, frozen** `data/splits/train.csv` and `data/splits/val.csv`; do not resplit by model.
5. Prepare the selected local image cache and verify the data pipeline:

   ```powershell
   python data/prepare_images.py
   python -m unittest discover -s tests -p "test_data.py" -v
   ```

6. Audit and split-generation implementation is available in `data/audit_images.py`, `data/check_leakage.py`, and `data/make_splits.py`. Do not overwrite frozen manifests or open the official TEST labels for development.

Each training run records its config and a split fingerprint. Final model training and evaluation must use the same fixed split and preprocessing rules across architecture families; report validation results separately from final official TEST results.

## 6. Scope, bias, and evaluation limitations

The collection is drawn from real cheque CAR regions, so it is more realistic than clean synthetic digits but is **not representative of all handwriting**. Results may vary with background noise, image quality, source bank, handwriting style, string length, and digit frequency. The observed reference-digit distribution is not balanced. The project reports corpus character error rate (**CER**), exact-string match, per-digit sequence-aligned metrics and error types. These project metrics should **not be presented as identical to the original ICFHR competition's ANLD metric**.

For the project-wide comparison, validation is used for tuning and model selection. The original official TEST partition is evaluated only against frozen final checkpoints; it is never used to tune or select models. Archived M4 Official TEST aggregates are documented in `docs/M4_Official_Test_Evidence.md`; keep M1-M3 Official TEST results separate until their evaluations are complete and audited.

---

**Documentation evidence:** `data/audit_outputs/*.txt`, `data/splits/*.csv`, `src/datasets/*.py`, `data/make_splits.py`, and the ORAND-CAR-2014 official dataset page. This document describes the dataset pipeline; M1 model results and failure analysis are reported separately in `Baseline_Analysis.md`.
