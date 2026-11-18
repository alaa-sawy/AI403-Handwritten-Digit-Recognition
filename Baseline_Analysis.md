# AI403 - M1 Baseline: Validation Results and Error Analysis

**Status:** Baseline tuning and three final-seed runs completed. **All metrics in this document use the frozen validation split, NOT the official test set.**

## 1. Scope and data protocol

- **Task:** Transcription of handwritten digit strings from ORAND-CAR-2014 (CAR-A and CAR-B).
- **Official training data:** 5,009 entries; one exact duplicate excluded, leaving 5,008 unique entries for the frozen development split.
- **Development split:** 4,507 training samples and 501 validation samples.
- **Official test set:** 6,710 samples; kept sealed for final evaluation only. No test metrics are reported here.
- **Input:** Grayscale images normalized to height 48 with aspect ratio preserved, with padding for batching.
- **Output:** Variable-length digit strings; CTC blank index 0, digits 0-9 mapped to indices 1-10.
- **Limitations:** Exact file/hash overlap was checked; writer-independent split cannot be asserted without writer IDs. Perceptual near-duplicate checks were not established by this audit.

## 2. Baseline architecture and protocol

The baseline is a small CNN with three convolutional stages followed by height-only pooling, a linear per-time-step classifier, and a CTC objective with greedy decoding. It has **94,091 trainable parameters**.

- Shared optimizer: Adam; batch size 32; weight decay 0.0001; gradient clipping 5.0.
- Maximum epochs: 120; early-stopping patience: 20, based on validation CER with validation CTC loss as the tie-breaker.
- Hyperparameter tuning: six configurations, comprising learning rates {0.001, 0.0003, 0.0001} crossed with dropouts {0.1, 0.3}, using seed 403.
- Best tuning configuration: learning rate **0.001**, dropout **0.1**, augmentation enabled. Best tuning validation CER: **0.782709**.
- Independent final training seeds: **7, 21, 42**, with frozen preprocessing and data splits.

The tuning result and the three final-seed results must be distinguished. The tuned seed is not a fourth final replicate.

## 3. Final validation results

| Seed | Best epoch | Validation CER | Exact match | Training seconds | Model-forward ms/image |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 19 | 0.783867 | 0.0000 | 1,299.57 | 1.237 |
| 21 | 10 | 0.782323 | 0.0000 | 2,589.55 | 0.340 |
| 42 | 28 | 0.784639 | 0.0000 | 985.45 | 0.505 |
| **Mean +/- sample SD** | - | **0.783610 +/- 0.001179** | **0.0000 +/- 0.0000** | **1,624.86 +/- 850.09** | **0.694 +/- 0.477** |

The final **validation CER is 78.361% +/- 0.118 percentage points**, and validation exact-string match is **0.0% for all three seeds**. These scores establish a reproducible but weak baseline; they are not official test scores.

## 4. Error decomposition

The evaluation uses character-level edit-distance alignment to count deletions, substitutions and insertions.

| Edit type | Seed 7 | Seed 21 | Seed 42 | Total across three evaluations | Share of edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deletions | 1,591 | 1,591 | 1,589 | **4,771** | **78.33%** |
| Substitutions | 440 | 436 | 444 | **1,320** | **21.67%** |
| Insertions | 0 | 0 | 0 | **0** | **0.00%** |
| **Total edits** | **2,031** | **2,027** | **2,033** | **6,091** | **100%** |

Each evaluation concerns the **same 501 validation samples**. The combined counts above are repeated-evaluation edit counts, not errors from 1,503 independent images. There are 2,591 reference characters per evaluation (7,773 across three evaluations).

### Prediction concentration

| Seed | Predictions equal to `10` | Fraction of 501 validation strings |
| ---: | ---: | ---: |
| 7 | 493 | 98.40% |
| 21 | 498 | 99.40% |
| 42 | 486 | 97.01% |
| **Total** | **1,477 of 1,503 predictions** | **98.27%** |

This strong concentration of decoded outputs on `10`, coupled with widespread deletions and zero exact matches, is evidence of **prediction collapse** rather than successful digit-string transcription. The underlying causal mechanism has not been established and should not be attributed exclusively to the CNN, CTC objective, sequence lengths or dataset without additional evidence.

### Real validation error examples

| Reference | Prediction | Edit distance |
| --- | --- | ---: |
| `540` | `10` | 2 |
| `100` | `10` | 1 |
| `109` | `10` | 1 |
| `10000` | `10` | 3 |

For the course report's error-analysis table, the most defensible interpretation is:

| Error type | Count, all three seed evaluations | Illustrative reference / prediction | Evidence-based interpretation |
| --- | ---: | --- | --- |
| Deletion | 4,771 | `10000` / `10` | Short decoded strings omit many reference digits. The underlying cause of that short decoding is not proven. |
| Substitution | 1,320 | `540` / `10` | Decoded symbols are concentrated on a small vocabulary, so many reference digits are represented incorrectly. Individual edit alignments can be ambiguous. |
| Insertion | 0 | No insertion observed | There is no evidence of excessive output length in these validation runs. |

*These counts combine repeated evaluations on the same 501 validation images, not 1,503 independent samples. The example pairs illustrate failures; they do not claim that every edit in an example is of a single type.*

## 5. Per-digit analysis

The metrics below are derived from character-level alignment, and therefore describe aligned digit detection, **not exact-position classification**. Support is identical across seeds because the validation split is frozen.

| Digit | Validation support | F1 seed 7 | F1 seed 21 | F1 seed 42 | Mean F1 |
| :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 940 | 0.470 | 0.476 | 0.467 | **0.471** |
| 1 | 290 | 0.559 | 0.559 | 0.559 | **0.559** |
| 2 | 201 | 0.000 | 0.000 | 0.000 | **0.000** |
| 3 | 185 | 0.000 | 0.000 | 0.000 | **0.000** |
| 4 | 185 | 0.000 | 0.000 | 0.000 | **0.000** |
| 5 | 207 | 0.019 | 0.010 | 0.036 | **0.022** |
| 6 | 160 | 0.000 | 0.000 | 0.000 | **0.000** |
| 7 | 149 | 0.000 | 0.000 | 0.000 | **0.000** |
| 8 | 149 | 0.000 | 0.000 | 0.000 | **0.000** |
| 9 | 125 | 0.000 | 0.000 | 0.000 | **0.000** |

Additional interpretation:

- Digit **1** has precision 0.441 and recall 0.762 in each seed. Its reasonably high recall is accompanied by many false positives, consistent with near-constant prediction of `10`.
- Digit **0** has recall around 0.354-0.364; it is frequently omitted despite being the most common reference digit (940 of 2,591).
- Digits **2, 3, 4, 6, 7, 8, and 9** have zero aligned true positives in all three evaluations.
- Digit **5** is nearly never recognized. The precision of 1.0 reported in seed 21 is misleading in isolation: recall is only about 0.005 (approximately **one true positive out of 207**).

The prediction-collapse observation is well supported by repeated seeds. Whether this stems primarily from capacity, CTC dynamics, sequence handling, optimization or data characteristics is a hypothesis, not a proven causal conclusion.


## 6. Sequence-length analysis

The validation predictions saved by the three final checkpoints were compared with their respective reference-label lengths. This is a **post-hoc analysis** of the frozen validation predictions, not a new training run.

| Metric | Seed 7 | Seed 21 | Seed 42 |
| --- | ---: | ---: | ---: |
| Mean reference length (digits) | 5.17 | 5.17 | 5.17 |
| Mean decoded prediction length (digits) | 2.00 | 2.00 | 2.00 |
| Decoded prediction shorter than reference | 499 / 501 | 499 / 501 | 499 / 501 |
| Fraction shorter than reference | 99.60% | 99.60% | 99.60% |

Thus, **sequence under-prediction is systematic**, rather than a few outliers. The same 501 images are reused across seeds; the three columns are model-run replications, not independent evaluation datasets. The length deficit is consistent with the deletion-dominated edit decomposition in Section 4.

## 7. CTC frame-level blank analysis (final Seed 7 checkpoint)

The saved best checkpoint from Seed 7 (best epoch 19) was loaded **without changing its weights**. For each validation image, the most likely token (`argmax`) was counted over its **valid output timesteps only**, excluding padded positions.

| Frame-level argmax token | Count | Meaning |
| --- | ---: | --- |
| Token 0 | 79,017 | CTC blank |
| Token 2 | 501 | Digit `1` |
| Token 1 | 493 | Digit `0` |
| Token 6 | 6 | Digit `5` |
| All other tokens | 0 | No frame-level argmax selections |
| **Total valid timesteps** | **80,017** | **501 validation images** |

The **blank-argmax fraction** was **79,017 / 80,017 = 98.75%**. This quantity is **not** the mean posterior probability of the blank symbol: it is the fraction of timesteps whose highest-scoring token was blank.

Under standard CTC greedy decoding, consecutive repeated tokens are collapsed and blank tokens removed [1]. Therefore, the observed blank-dominated and highly restricted frame-level argmax outputs are consistent with the mostly two-digit decoded predictions and extensive deletions. **A high blank proportion alone is not evidence of incorrect CTC training**, because blank-heavy frame alignments can also arise in working CTC systems. Our evidence for failure comes from the combination of blank dominance, the restricted nonblank vocabulary, nearly constant decoded strings, length mismatch and measured CER. This diagnostic was performed for Seed 7 only; it must not be labelled a three-seed mean.

## 8. Training dynamics: CTC loss versus decoded CER

Inspection of the saved **Seed 7 training history** shows that CTC loss and sequence-level CER did not improve together at the end of training:

| Metric | Epoch 32 | Epoch 39 | Direction |
| --- | ---: | ---: | --- |
| Training CTC loss | 2.1419 | 2.1103 | Decreased |
| Validation CTC loss | 2.1611 | 2.1041 | Decreased |
| Validation CER | 0.7850 (78.50%) | 0.9089 (90.89%) | Increased (worsened) |

CTC optimizes the summed probability of possible frame-to-label alignments [1], whereas CER measures edit distance after the chosen decoding procedure. They are related but distinct quantities. In these epochs, lower CTC loss did **not** yield better greedy-decoded transcripts. This pattern does **not**, by itself, establish classic overfitting because validation CTC loss also decreased. The final reported Seed 7 validation CER remains **0.783867**, measured from the **best checkpoint at epoch 19**, not the last training epoch.

## 9. Scientific interpretation and causal limits

**Established by the saved evaluation artifacts:**

1. **Prediction collapse:** Nearly every validation image was decoded as `10` (Section 4).
2. **Systematic under-prediction:** 499 of 501 predictions per seed were shorter than their references (Section 6).
3. **Restricted CTC emissions (Seed 7):** Blank was the frame-level argmax at 98.75% of valid timesteps; only digits `0`, `1` and `5` appeared as nonblank argmax tokens (Section 7).
4. **Loss/metric divergence:** Validation CTC loss decreased while greedy-decoded CER worsened over epochs 32-39 for Seed 7 (Section 8).

**Plausible but not proven root-cause hypotheses:** The CNN baseline has no explicit recurrent or attention-based sequence context module. CRNN research provides a theoretical motivation for coupling convolutional visual features with recurrent sequence modeling [2], but the present observations **do not isolate the absence of recurrence as the cause**. The optimization and alignment behavior of CTC could also contribute [1]. Comparing the Classic, Modern and Advanced architectures under the frozen protocol may provide further evidence, but such comparisons must not be represented as controlled causal proof of one mechanism when several components change simultaneously.

In particular, we **do not** conclude that the dataset labels are faulty, that CTC itself is defective or that the model overfits, because those claims have not been established by the analyses performed here.

## 10. Computational cost

| Metric | Seed 7 | Seed 21 | Seed 42 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Training time (minutes) | 21.66 | 43.16 | 16.42 | Mean 27.08 +/- 14.17 minutes |
| Model-forward time (ms/image) | 1.237 | 0.340 | 0.505 | Mean 0.694 +/- 0.477; excludes image loading and preprocessing |
| Observed peak PyTorch allocated GPU memory (MiB) | 615.78 | 615.78 | 615.78 | Not total GPU memory usage |
| Observed process CPU RSS (MiB) | 1,627.54 | 4,019.96 | 2,961.89 | Sampled process memory, not whole-system RAM |
| Parameter count | 94,091 | 94,091 | 94,091 | Trainable model parameters |
| Saved checkpoint size (MiB) | 0.363 | 0.363 | 0.363 | `best.pt`; more precise ~0.362704 MiB |

**Training hardware:** NVIDIA RTX A1000 Laptop GPU (4 GB VRAM), Windows 11, for the reported final runs.

These measurements serve as a cost baseline for later architectures. Wall-clock comparisons across different machines must disclose the hardware, runtime environment and batch-size differences; raw timing alone cannot establish architectural efficiency.

## 11. Conclusions and reporting limits

1. The baseline was trained and evaluated through a common, logged protocol and three independent final seeds.
2. Its final validation CER is consistently poor, with no correctly transcribed complete validation strings.
3. Error analysis shows that deletions dominate and decoded predictions have collapsed almost entirely to `10`; validation outputs are shorter than their references in 99.60% of cases per seed.
4. Per-digit analysis corroborates the failure to distinguish most digit classes. Blank-argmax analysis of the Seed 7 checkpoint and saved training curves provide additional evidence about decoding behavior, but do not establish a unique root cause.
5. These findings motivate comparison with the Classic, Modern and Advanced architectures **under the same frozen evaluation protocol**, without altering the baseline retrospectively merely to improve its score.
6. Official test labels remain sealed; validation results must not be presented as test results.

## 12. Reproducibility artifacts (local repository)

- `configs/baseline.json` and `configs/protocol.json`
- `experiments/baseline_results/tuning_results.csv`
- `experiments/baseline_results/best_tuned_config.json`
- `experiments/baseline_results/final_runs.csv`
- `experiments/baseline_results/final_mean_sd.json`
- `experiments/baseline_results/final/seed_7/`, `seed_21/`, `seed_42/`: run summaries, histories, curves, validation predictions, per-digit metrics, substitution confusions and error examples
- `checkpoints/baseline_results/`: local best model weights (excluded from Git; share separately or provide reproduction instructions)

## 13. Academic references

[1] Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.* In **Proceedings of the 23rd International Conference on Machine Learning (ICML)**, 369-376. https://doi.org/10.1145/1143844.1143891

[2] Shi, B., Bai, X., & Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition.* **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 39(11), 2298-2304. https://doi.org/10.1109/TPAMI.2016.2646371

*All empirical results in this document come from the local team's frozen validation split and logged experiments. The references support the theoretical discussion; they are not sources of the reported empirical measurements.*
