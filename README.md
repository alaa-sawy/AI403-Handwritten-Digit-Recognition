# AI403 - Handwritten Digit-String Recognition

**Pharos University in Alexandria | Faculty of Computer Science & Artificial Intelligence | Deep Learning (AI403), Fall 2026**  
**Instructor:** Dr. Sahar Ghanem  
**Selected course idea:** #12 - Handwritten Text Recognition, specialized to handwritten **digit strings** from bank-cheque courtesy-amount fields.

**Repository:** https://github.com/alaa-sawy/AI403-Handwritten-Digit-Recognition  
**Checkpoint / dataset transfer Drive:** https://drive.google.com/drive/folders/1y6Tq6SnGyZ_oa1Vexl4Ovsx27PQArUZL?usp=sharing

> **Verified repository status (20 November 2026):** Five primary models have six-trial tuning and three-seed **validation** evidence. Modern Augmentation ON/OFF and Advanced No-Multiscale studies are also recorded. M4 has separately audited archived **Official TEST** results. M1-M3 Official TEST comparison is **not yet committed as a complete table**. M5 has committed Streamlit demo source, but the currently missing `src/inference.py` and missing `streamlit` dependency block a verified clean-clone demo launch. Final team PDF/slides and signed contribution records are not yet present on `main`. See the [submission checklist](#submission-checklist-and-known-blockers).

## README map - course requirements

This README follows the **nine required README items in section 4.2** of the AI403 project brief:

1. [Project, members, and chosen idea](#1-project-members-and-chosen-idea)
2. [Task description and best measured result](#2-task-description-and-best-measured-result)
3. [Environment setup](#3-environment-setup)
4. [Dataset download and preprocessing](#4-dataset-download-and-preprocessing)
5. [Training commands for every model](#5-training-commands-for-every-model)
6. [Evaluation commands, metrics, and controlled studies](#6-evaluation-commands-metrics-and-controlled-studies)
7. [Live demo command and present limitations](#7-live-demo-command-and-present-limitations)
8. [Hardware and expected historical runtimes](#8-hardware-and-expected-historical-runtimes)
9. [Credits, papers, pretrained components, and licence](#9-credits-papers-pretrained-components-and-licence)

## 1. Project, members, and chosen idea

| Member | Student ID | Assigned workstream |
| --- | --- | --- |
| Alaa Abdelhalim Abdelrady Elsawy | 202304181 | **M1:** dataset, preprocessing, shared trainer/evaluator, CNN-CTC baseline |
| Dalia Mahmoud Ahmed Mahmoud Arfa | 202303181 | **M2:** Classic ResNet18 scratch and pretrained + BiGRU + CTC |
| Mariam Adel Mohamed Abdelmeguid | 202303361 | **M3:** Modern CNN-Transformer-CTC and augmentation ON/OFF |
| Salama Ahmed Salama Hassan Ahmed | 202302079 | **M4:** Advanced MultiScale CNN-BiGRU-attention and ablation |
| Kerolos Elromany Naeem Ragheb Mashriqi | 202302674 | **M5:** shared comparison, integration, visualization, Streamlit demo |

**Official team name:** to be confirmed by the team before using the submission filename `DL_Project_TeamName.zip`. The project title above is not a substitute for a registered team name.

**Input:** an image of a cropped handwritten numeric amount on a cheque.  
**Output:** the complete digit **string** (e.g., `00520`); leading zeros, repeated digits and their order are significant.

The project recognizes variable-length strings without manual character segmentation. All five primary model variants use **Connectionist Temporal Classification (CTC)**, with blank class ID 0 and digits 0-9 encoded as IDs 1-10. Inputs are grayscale tensors `[B,1,48,W]`; model outputs are frame logits and their valid output lengths.

**Research hypotheses (stated before final TEST interpretation):**

- **H1:** contextual sequence models perform better than a simple CNN-CTC reference. The Classic/Advanced validation results support an architecture-family difference, but do not isolate recurrence as the only cause.
- **H2:** a Transformer encoder offers a viable alternative to GRU-based models. Modern works, though its recorded validation CER is higher than Classic Pretrained and Advanced.
- **H3:** a custom MultiScale block improves the Advanced network. Matched three-seed **validation** means were essentially identical with and without it, so no clear benefit was established.
- **H4:** training-only augmentation improves Modern generalization. Mean Modern **validation** CER was slightly better with ON, but seed effects were mixed.

## 2. Task description and best measured result

The data are the publicly documented **ORAND-CAR-2014 CAR-A and CAR-B** digit-string images. The five primary architectures are:

| Registered model | Architecture | Parameters | Initialization |
| --- | --- | ---: | --- |
| `baseline` | Lightweight CNN + CTC | 94,091 | Scratch |
| `classic_scratch` | Grayscale-adapted ResNet18 + 2-layer BiGRU + CTC | 13,541,323 | Scratch |
| `classic_pretrained` | Same ResNet18 + BiGRU + CTC | 13,541,323 | ImageNet ResNet18 backbone |
| `modern` | CNN + 2-layer / 4-head Transformer encoder + CTC | 1,183,179 | Scratch |
| `advanced` | Custom MultiScale CNN + BiGRU + multi-head attention + CTC | 1,245,867 | Scratch |

**Primary three-seed VALIDATION results** (501 samples, mean +/- sample SD; standard deviation is in percentage points):

| Model | CER (%) - lower is better | Exact-string match (%) - higher is better |
| --- | ---: | ---: |
| Baseline | 78.361 +/- 0.118 | 0.000 +/- 0.000 |
| Classic Scratch | 3.242 +/- 0.253 | 85.828 +/- 0.998 |
| Classic Pretrained | 2.676 +/- 0.161 | 88.090 +/- 0.756 |
| Modern (augmentation ON) | 4.052 +/- 0.241 | 83.367 +/- 1.170 |
| **Advanced Full** | **2.316 +/- 0.306** | **89.621 +/- 1.584** |

**Best recorded mean validation CER:** Advanced Full, **2.316%**. This is a validation/model-selection finding, **not** the final five-model TEST ranking. Baseline is included as the required reference model; its very high error rate and 0% whole-string exact match make it unsuitable as the deployed recognizer.

**Separately audited M4 Official TEST** (6,710 samples per seed; these are **TEST**, not validation):

| M4 condition | TEST CER (%) | TEST exact-string match (%) |
| --- | ---: | ---: |
| Advanced Full | **2.4981 +/- 0.3520** | **89.5181 +/- 1.3729** |
| Advanced No-Multiscale | 2.6055 +/- 0.2495 | 89.1207 +/- 0.9222 |

Source: [M4 Official TEST audit and checkpoint hashes](docs/M4_Official_Test_Evidence.md). The archived M4 files passed independent saved-output consistency checks; the original evaluation invocation / Git revision is not independently reconstructed. **Do not tune or select checkpoints on TEST.** The difference does not prove a reliable causal MultiScale advantage.

## 3. Environment setup

The project was developed and evaluated on **Windows 11**, Python **3.13/3.14**, PyTorch **2.15.0+cu132**, TorchVision **0.29.1+cu132**, and NVIDIA GPUs. The committed `requirements.txt` pins those PyTorch versions for Windows/Linux; install using a compatible Python environment with access to the indicated wheels. Check package compatibility on other platforms before running.

**PowerShell from a clean local folder:**

~~~powershell
git clone https://github.com/alaa-sawy/AI403-Handwritten-Digit-Recognition.git
cd AI403-Handwritten-Digit-Recognition
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
& $py -m unittest discover -s tests -v
~~~

If `torch` import fails, **fix the interpreter/environment first**; a successful virtual environment creation alone does not mean model inference is ready. CUDA is optional for ordinary inference, but GPUs accelerate large-model training. Do not run tuning/final training again simply to reproduce saved tables.

**Repository organization (selected files):**

~~~text
configs/           protocol and per-model base configurations
data/              dataset documentation, audit, split, preprocessing scripts
src/datasets/      loaders, preprocessing and augmentation
src/models/        baseline, classic, modern, advanced, registry
src/train.py       shared trainer
src/evaluate.py    shared evaluator for validation / final TEST
src/experiments_runner.py  tuning and final-seed orchestration
experiments/       CSV/JSON, plots, summaries, error analysis
checkpoints/       local checkpoints (large files normally supplied through Drive)
m5/ and shared/    comparison script and shared utility wrappers
demo/app.py        M5 Streamlit frontend (integration blocker described below)
docs/              technical model/data evidence and M4 materials
contributions/     per-student records / review templates
report/            intended final report and slides (not yet committed)
~~~

The shared final trained checkpoint is identified by its **model name + seed + saved config**. Download weights from the linked Drive and restore the folder structure before inference; checkpoint paths inside historical JSON can refer to the original contributor's laptop.

## 4. Dataset download and preprocessing

**Dataset:** [ORAND-CAR-2014 / ICFHR 2014 HDSRC](https://www.orand.cl/icfhr2014-hdsr/). [Documentation and licence](data/DATASET.md). The raw original archive is not redistributed in Git.

| Partition | CAR-A | CAR-B | Total | Purpose |
| --- | ---: | ---: | ---: | --- |
| Official original TRAIN | 2,009 | 3,000 | 5,009 | Development source |
| Frozen project TRAIN | 1,808 | 2,699 | **4,507** | Weight updates |
| Frozen project VALIDATION | 201 | 300 | **501** | Tuning / checkpoint selection / error analysis |
| Official TEST | 3,784 | 2,926 | **6,710** | Evaluation only after freezing decisions |

One exact duplicate in the original training entries was excluded from the development manifest: the original dataset was not altered. Frozen split seed `403`; stratification used CAR source and sequence length. There were **no exact image-hash train/validation or TRAIN/TEST collisions** in the recorded checks; writer-disjointness and perceptual near-duplicate absence have **not** been proven.

**Download original archive and verify identity:**

~~~powershell
New-Item -ItemType Directory -Force -Path .\data\raw | Out-Null
Invoke-WebRequest "https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz" -OutFile .\data\raw\ORAND-CAR-2014.tar.gz
Get-FileHash .\data\raw\ORAND-CAR-2014.tar.gz -Algorithm SHA256
~~~

Expected SHA-256 (original archive):

~~~text
7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a
~~~

The frozen `data/splits/train.csv` and `val.csv` are already committed. **Do not regenerate or overwrite these manifests** when using the saved experiments. Prepare only their local image cache:

~~~powershell
& $py .\data\prepare_images.py
& $py -m unittest discover -s tests -p "test_data.py" -v
~~~

Preprocessing: white-composite transparent pixels, grayscale, aspect-ratio-preserving resize to height **48**, normalize to `[0,1]`, pad **on the right** per batch and track valid widths. Training-only augmentation uses roughly **+/-2 degree rotation** and brightness **0.90-1.10** when enabled. **Validation/TEST images are never augmented**. The official TEST images/labels are not published in this repository.

## 5. Training commands for every model

This section documents **reproduction commands**, not work that still needs to run for final submission. The experiment protocol used Adam, CTC loss, batch size 32, weight decay 0.0001, gradient clip 5.0, maximum 120 epochs, patience 20, and the same **six-point tuning grid** for every primary model (learning rates `0.001`, `0.0003`, `0.0001` crossed with dropout `0.1`, `0.3`).

**Tuning seed:** `403`. **Independent final seeds:** `7, 21, 42`. Tuning chooses the lowest validation CER, then the chosen configuration is trained independently on each final seed.

Run from the repository root with the prepared local training/validation cache:

~~~powershell
# Baseline (M1)
& $py -m src.experiments_runner --model baseline --stage tune
& $py -m src.experiments_runner --model baseline --stage final

# Classic Scratch (M2)
& $py -m src.experiments_runner --model classic_scratch --stage tune
& $py -m src.experiments_runner --model classic_scratch --stage final

# Classic Pretrained (M2)
& $py -m src.experiments_runner --model classic_pretrained --stage tune
& $py -m src.experiments_runner --model classic_pretrained --stage final

# Modern with training augmentation ON (M3)
& $py -m src.experiments_runner --model modern --stage tune
& $py -m src.experiments_runner --model modern --stage final

# Advanced Full (M4)
& $py -m src.experiments_runner --model advanced --stage tune
& $py -m src.experiments_runner --model advanced --stage final

# Paired studies, AFTER corresponding final runs:
& $py -m src.experiments_runner --model modern --stage aug-study
& $py -m src.experiments_runner --model advanced --stage ablation
~~~

Saved experiment outputs: `experiments/<model>_results/tuning_results.csv`, `best_tuned_config.json`, `final_runs.csv`, `final_mean_sd.json` and `final/seed_<N>/` with run summaries, loss/CER curves and validation-error files. **All 30 tuning and 15 five-model final runs, plus three Modern OFF and three No-Multiscale ablation runs, are recorded (51 scientific runs).** Historical run reuse checks absolute saved checkpoint paths; don't launch a new experiment over old files without deliberate isolation. The M4 `advanced_no_attention` variant is implemented but **not supported by three matched final experimental runs**.

## 6. Evaluation commands, metrics, and controlled studies

**Validation evaluation of a saved checkpoint** (writes beside `best.pt`; no training):

~~~powershell
& $py -m src.evaluate --checkpoint .\checkpoints\advanced_results\final\seed_7\best.pt --split val
~~~

**Final Official TEST — only with frozen checkpoints and the agreed evaluation protocol:**

~~~powershell
# One-time preparation on a lab that has NOT yet unsealed TEST:
& $py .\data\prepare_final_test.py --confirm-final

# Example official evaluation after TEST is prepared:
& $py -m src.evaluate --checkpoint .\checkpoints\advanced_results\final\seed_7\best.pt --split test --final-test
~~~

**Do not re-run the preparation script if `data/splits/test.csv` already exists**; it explicitly rejects overwriting. M1-M3 Official TEST was being collected on two dedicated lab machines after checkpoint-selection freeze. No complete audited M1-M3 TEST table was present in the reviewed GitHub snapshot; don't substitute their validation results as TEST results. Archived six-seed M4 TEST results are linked above. Any public summary should contain aggregate metrics, not raw TEST labels, images, `predictions.csv` or `errors.csv`.

**Task-appropriate metrics:** corpus CER = (substitutions + deletions + insertions) / reference characters; exact complete-string match; per-digit precision, recall, F1 and support; error type counts and CAR-subset/length breakdowns. Each reported final condition has 3 independently trained seeds and sample SD. Sample SD across seeds is **not** a confidence interval.

**Controlled experiments** (all values below are **validation**):

| Study | Compared conditions | CER (%) mean +/- SD | Interpretation |
| --- | --- | --- | --- |
| ImageNet initialization | Classic Scratch / Pretrained | 3.242 +/- 0.253 / **2.676 +/- 0.161** | Pretrained lower by 0.566 pp; selected LRs also differ |
| Augmentation | Modern ON / OFF | **4.052 +/- 0.241** / 4.194 +/- 0.226 | ON lower by 0.142 pp; individual seed effects mixed |
| Architectural ablation | Advanced Full / No-Multiscale | 2.316 +/- 0.306 / 2.316 +/- 0.204 | No demonstrated mean validation improvement |

Sources: [M2 final summaries](experiments/classic_pretrained_results/final_mean_sd.json), [Modern augmentation study](experiments/modern_results/augmentation_study.csv), [M4 ablation](experiments/advanced_results/ablation_results.csv), [Modern technical record](docs/M3_Modern_Results.md).

**Recorded VALIDATION error examples (not Official TEST):**

| Model (seed 7) | Reference | Prediction | Error |
| --- | --- | --- | --- |
| Baseline | `10000` | `10` | Multiple deletions |
| Classic Scratch | `15366` | `15866` | Digit substitution |
| Classic Pretrained | `23104` | `218` | Multiple edits |
| Modern | `2510` | `1510` | Digit substitution |
| Advanced | `2510` | `250` | Deletion |

The Baseline frequently collapses to `10` and has substantial deletion errors. Visual causes cannot be proven from numerical error tables alone. See [Baseline scientific error analysis](docs/Baseline_Analysis.md), [M3 error analysis](docs/M3_Modern_Results.md) and the per-seed `validation/` files. The Advanced padding audit [`experiments/padding_audit_results.json`](experiments/padding_audit_results.json) documents batch-padding sensitivity: **use the checkpoint's evaluation batch size** and avoid uncontrolled speed/performance comparisons.

## 7. Live demo command and present limitations

M5 committed a **Streamlit frontend** at [`demo/app.py`](demo/app.py), shared tools under [`shared/`](shared/) and validation comparison figures under [`m5/results/`](m5/results/).

**When the missing adapter and dependency are resolved**, run locally (the demo is intended to infer from downloaded, trained checkpoints, not an external AI API):

~~~powershell
& $py -m pip install streamlit
& $py -m streamlit run demo/app.py
~~~

**IMPORTANT: not yet verified working from a clean clone.** As of the inspected `main` revision:

- `demo/app.py` imports `src.inference` (`MODEL_CHECKPOINTS`, `checkpoint_for`, `load_checkpoint_model`, `predict_image`), but `src/inference.py` **does not exist** on `main`.
- `streamlit` is **not included** in `requirements.txt`, so the standard environment setup alone cannot launch the UI.
- M5's guide mentioned `tests/test_m5_integration.py`, but that test was **not found** on `main`. Required live inference with a real uploaded image and an actual final checkpoint is therefore **not independently verified**.

M5 must add the adapter, register the Streamlit dependency, restore usable checkpoints, run model integration tests and demonstrate one real local inference before the 5-minute course demo can be claimed complete. Do **not** treat the mere presence of the frontend as a working live demonstration. The default Baseline should not be presented as a reliable digit recognizer.

## 8. Hardware and expected historical runtimes

The values below are **observed historical training and model-forward inference measurements**, not guaranteed times on a new computer. Train/inference values are mean across 3 seeds; GPU memory is **seed-7 peak PyTorch allocation** (MiB), not overall VRAM in use. The hardware column identifies the seed-7 device, not a claim that all models ran on identical hardware.

| Model / condition | Train minutes (mean) | Forward ms/image (mean) | Parameters | Checkpoint MiB | GPU peak MiB (seed 7) | GPU (seed 7) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 27.1 | 0.694 | 94,091 | 0.36 | 615.8 | RTX A1000 Laptop |
| Classic Scratch | 63.8 | 3.503 | 13,541,323 | 51.73 | 1,987.8 | RTX A1000 Laptop |
| Classic Pretrained | 100.9 | 3.040 | 13,541,323 | 51.73 | 1,988.0 | RTX A2000 Laptop |
| Modern ON | 46.5 | 0.365 | 1,183,179 | 5.51 | 429.5 | RTX A1000 Laptop |
| Modern OFF | 14.9 | 0.438 | 1,183,179 | 5.51 | 446.6 | RTX A2000 Laptop |
| Advanced Full | 30.2 | 1.869 | 1,245,867 | 4.77 | 1,465.4 | RTX 3060 Laptop |
| Advanced No-Multiscale | 41.8 | 1.665 | 1,163,691 | 4.46 | 1,260.0 | RTX 3060 Laptop |

Different machines and interruptions make wall-clock/forward timing comparisons **descriptive**, not a controlled ranking. The **full demo latency** (reading an image + preprocessing + loading model + forward + decoding) is not measured by these recorded forward-only timings. Source values: `experiments/*/final/seed_*/run_summary.json` and the controlled-study per-seed records.

## 9. Credits, papers, pretrained components, and licence

The system trains and predicts using the team's locally saved PyTorch models. It does **not** call commercial AI inference APIs.

- **Dataset:** Diem et al., *ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets (HDSRC 2014)*; [official ORAND page](https://www.orand.cl/icfhr2014-hdsr/).
- **Dataset licence as published:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Attribute the publisher, comply with noncommercial/no-derivatives restrictions, and do not redistribute original or processed dataset images without appropriate permission.
- **CTC:** Graves et al. (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML.
- **CNN + sequence recognition:** Shi, Bai and Yao (2017), *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition*, IEEE TPAMI ([preprint](https://arxiv.org/abs/1507.05717)).
- **ResNet18 / pretrained ImageNet:** He et al. (2016), *Deep Residual Learning for Image Recognition*, CVPR ([paper](https://arxiv.org/abs/1512.03385)); [TorchVision ResNet18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- **Transformer:** Vaswani et al. (2017), *Attention Is All You Need* ([paper](https://arxiv.org/abs/1706.03762)).
- **Software:** PyTorch, TorchVision, NumPy, Pillow, Matplotlib, psutil; Streamlit is the planned M5 demo UI dependency. Custom preprocessing, architectures, attention, masking, CTC integration and experiment controls are in the repository; library components are adapted and cited.

## Submission checklist and known blockers

The course brief requires **12 deliverables**: documented source code, dataset documentation, final checkpoints/reproduction scripts, complete experiments, fair model comparison with hypothesis verdicts, error analysis, ablation, a **15-25-page technical report**, **15-minute team presentation**, **5-minute live demo**, full README and one **weekly co-signed contribution log plus peer evaluation per student**.

**Report structure** must have the prescribed **22 sections** from Abstract through References. The seven mandatory report tables are: dataset comparison; architecture comparison; hyperparameters; main results (mean +/- SD); ablation; error analysis (counts and examples); and computational cost (parameters, checkpoint size, training time, inference time, GPU memory and hardware).

| Hand-in item | Evidence/status on audited `main` |
| --- | --- |
| Five models; 6 trials each; 3 final validation seeds each | Recorded |
| Modern augmentation and M4 no-multiscale ablation | Recorded |
| M4 archived Official TEST metrics | Audited aggregates in [M4 evidence](docs/M4_Official_Test_Evidence.md) |
| M1-M3 Official TEST aggregates | Not yet present as a complete audited table; two-lab evaluation in progress |
| M5 demo | Frontend present, **blocked** pending `src/inference.py` and Streamlit dependency |
| Final report PDF and slides in `report/` | Not tracked in current repository |
| Large final checkpoint archive | Drive link provided; **contents not independently inventoried here** |
| Contribution logs / peer reviews | Review templates present; signed, personally confirmed final PDFs pending |
| Formal team name | Pending |
| Repository TA collaborator access | Must be confirmed by team outside this repository audit |

**Official submission, per course brief (`20 November 2026` deadline):**

- One member emails the supervising TA a ZIP named `DL_Project_TeamName.zip`.
- ZIP should contain `TeamName_Report.pdf`, `TeamName_Slides.pdf` (or PPTX), and a text file with the GitHub URL. Include the signed contribution/peer records using the TA's required mechanism.
- Add the TA as a GitHub collaborator before the deadline. Put original datasets and large model checkpoints in shared Drive and link it here.
- Slides: **15 minutes** total across all five members; live demo: **5 minutes**. Every member should be ready for an individual viva and be able to explain the full system.
- Individual PDF naming: `TeamName_StudentID_Log.pdf`; each log must have **Week, Task, Hours, Evidence, Problems** and a teammate co-signature.

**Contribution source material:** [M1 log draft](contributions/M1_202304181_Log_DRAFT.md), M2-M5 draft logs and peer templates under [`contributions/`](contributions/). Git/experiment evidence supports technical tasks but does not replace the students' own hour records, peer evaluations or signatures.

---

**Reporting rule:** Validation results must never be relabelled as Official TEST. Do not invent missing scores, GPU metrics, experiment dates, signed logs or working-demo evidence. Work on training and final TEST is separate; freeze model selection before evaluating TEST.
