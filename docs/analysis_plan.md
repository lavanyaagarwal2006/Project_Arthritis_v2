# Statistical analysis plan (pre-registered)

Status: DRAFT v0.1. Freeze this file (git tag `sap-v1.0`) **before** any model is trained.
After the freeze, changes must be logged in the Amendments section with a date and a reason.

## 1. Intended use

A **decision-support aid** that flags joints with likely bone erosion on PA hand
radiographs. Erosion is the radiographic hallmark of erosive rheumatoid arthritis. There are
two use cases. They carry different risk, so each is validated separately.

**Use case A: monitoring by rheumatologists (primary for v2).**
- The specialist compares serial radiographs of a known RA patient and looks for new or
  progressing erosions, especially minimal (SvdH 1) ones.
- The specialist remains the decision-maker, and the tool is a second reader.
- Requirements:
  - per-joint erosion detection with calibrated probabilities
  - Phase 2: paired-visit change detection (new erosion at visit 2 relative to visit 1),
    using RAM-H1200 follow-up images where timepoints are available
- Validated by Sections 3–7 below.

**Use case B: triage by non-specialist physicians (future; higher risk).**
- A general physician, e.g. at a rural clinic or health camp, has a patient with suspected
  inflammatory arthritis and a hand radiograph, but no rheumatology training.
- The output supports a **refer-to-rheumatologist** decision. It never supports a diagnosis.
- This is higher risk because the user cannot catch the model's errors (automation bias).
  The images also differ from the development data: other equipment, sometimes phone
  photos of film, and a different population.
- Requirements before any such use:
  - external validation on Indian data
  - a high-sensitivity operating point (≥ 95%)
  - a prospective study in that setting against a rheumatologist reference standard, with consent
  - regulatory review (CDSCO Software as a Medical Device in India)
- Nothing in v2 is validated for use case B.

**Applies to both:**
- **Not for:** standalone diagnosis of RA, use without clinician review, pediatric patients,
  or non-hand radiographs.
- **Why erosion?** RA is diagnosed clinically (2010 ACR/EULAR criteria: joint involvement,
  serology, acute-phase reactants, symptom duration). Radiographs contribute through
  erosions. Erosions typical of RA are by themselves sufficient for RA classification
  under the 2010 criteria. Small, early erosions (SvdH score 1) are the ones human
  readers miss most often. This tool targets that gap.
- **Known blind spot:** pre-erosive early RA has normal radiographs. A negative output never
  rules out RA.

## 2. Data

- **Development data:** RAM-H1200 (Hugging Face `TokyoTechMagicYang/RAM-H1200-v1`).
  - 1,200 posteroanterior hand radiographs from 291 patients, collected at 6 centers in Sapporo, Japan.
  - Includes joint-level Sharp/van der Heijde (SvdH) bone erosion (BE, 0–5) and joint
    space narrowing (JSN, 0–4) scores.
  - License: CC BY-NC-SA 4.0 (non-commercial).
- **Unit of analysis:**
  - Joint ROI crop (16 SvdH erosion joints per hand).
  - Hand image (aggregate of its joints).
  - Patient (the unit of independence for all CIs).
- **Exclusions:** none planned. Any exclusion, and the reason for it, is logged in
  `splits/exclusions.csv`.

## 3. Endpoints

Erosion is coded as present when SvdH BE ≥ 1.

| Endpoint | Definition |
|---|---|
| **Primary** | Joint-level erosion detection: AUROC for erosion present (BE ≥ 1) vs absent (BE = 0) |
| **Key secondary** | Hand-level "any erosion": the hand is positive if any joint has BE ≥ 1. The hand score is the maximum of the joint probabilities. |
| **Early-detection secondary** | Joint-level AUROC restricted to BE = 1 (minimal erosion) vs BE = 0, i.e. the subtle cases |
| Other secondary | AUPRC; sensitivity at 90% specificity; specificity at 90% sensitivity; sensitivity, specificity, PPV and NPV at the locked threshold; expected calibration error (ECE, 15 bins) raw and after recalibration |

**Decision rule allowed before training:** after inspecting the *label distribution only*
(never model outputs), the primary and key secondary endpoints may be swapped if hand-level
negatives number fewer than 30 hands. In that case hand-level AUROC is too unstable to be
the primary endpoint. Record the decision in Amendments.

## 4. Validation design

1. **Sealed test set.** The official RAM-H1200 test split (patient-level) is not touched
   until the final model and threshold are locked. It is evaluated once.
2. **Internal validation.** Patient-grouped, outcome-stratified 5-fold CV on the official
   train+val splits. Out-of-fold predictions are used for model selection, the threshold
   and temperature fitting.
3. **Internal-external validation.** Leave-one-center-out (LOCO) on train+val. This shows
   how performance changes across sites. It is reported as "internal-external",
   **never as external validation**.
4. **External validation (future).** A cohort from a different institution and country.
   Candidates are Taipei Veterans General Hospital and ISIC New Delhi (Dr. Malviya, pending
   ethics approval). Only this tier is described as "external validation".

## 5. Models (each compared against the rung below)

1. Majority / prevalence baseline (AUROC 0.5 by construction; reported for the operating point)
2. Small 4-block CNN trained from scratch (no pretraining)
3. ResNet-50 (ImageNet), fine-tuned
4. DenseNet121 (ImageNet), fine-tuned
5. ConvNeXt-Tiny / EfficientNet-B0 (ImageNet), fine-tuned

Model selection uses mean out-of-fold AUROC only, never the test set.

## 6. Threshold and calibration

- The operating threshold is chosen on out-of-fold predictions to reach **≥ 90% joint-level
  sensitivity** (a screening aid should miss few erosions). It is then locked.
- Recalibration is Platt scaling, i.e. logistic recalibration with slope and intercept,
  cross-fitted across folds. Temperature scaling alone cannot undo the prevalence shift that
  balanced sampling introduces, so it is reported only as a comparison.

## 7. Statistics

- 95% CIs come from a **patient-clustered bootstrap**: 2,000 resamples of patients, with all
  of each patient's images and joints kept together.
- Model comparisons use the paired patient-clustered bootstrap difference in AUROC.
  The comparison is significant if the 95% CI excludes 0.
- No p-value fishing. Every metric listed above is reported for every model, including
  the ones that look bad.

## 8. Known limitations (to be reported)

- All development data comes from one city and one population (Japanese adults).
  Generalization to Indian or US populations is unknown until external validation.
- Phase 1 uses **annotated joint ROIs**. End-to-end performance with automatically
  detected joints will be lower and is measured in Phase 2.
- SvdH erosion scoring has known inter-reader variability, so the labels are noisy.
- A non-commercial data license applies.

## 9. Reporting

Results are reported following the TRIPOD+AI and CLAIM 2024 checklists.

## Amendments

**A1 (2026-10-11, before freeze; based on labels and metadata only, no model outputs).** The
dataset was inspected with `make inspect`, and the plan was adjusted as follows.

- **Erosion scale.** Observed per-joint erosion scores are {0, 1, 2, 3, 5}, which is the SvdH
  hand erosion scale 0–5 with no 4s present. The binary endpoint (score ≥ 1) is unchanged. Any
  ordinal model uses the observed levels.
- **Hand-level negatives.** There are 170 hands with no erosion (146 train, 12 val, 12 test),
  above the 30-hand rule in Section 3, so the joint-level endpoint stays primary.
- **Patient key.** `Normalized PatientID` restarts its numbering in each center: file names carry
  the center plus the same number. The patient key is therefore `center + PatientID`, decided
  automatically by `manifest.py` and asserted in a unit test. Matching on the number alone would
  merge different people from different centers.
- **Metadata unit.** Metadata is per study (`Mapped Image Stem`, without the `_L`/`_R` suffix),
  and each study can contribute both hands. Both hands of a patient always share a fold.
- **Leave-one-center-out.** Held-out centers are HMCRD, SARC and SCGH. HU1–3 (30 hands in total)
  are train-only, set by `loco_min_hands: 50`.
- **Pixel spacing differs by center.** It is 0.175 mm in a group matching SARC's count and 0.15 mm
  in the rest. This is a possible shortcut, because a model can learn to recognise the center.
  Leave-one-center-out measures how much it matters, and resampling crops to a common mm/pixel
  scale is planned (Methods #1).
- **Subgroups to report.** Sex (F 746 / M 78 / O 12 studies), age band, hand side, and follow-up
  visit vs first visit (StudyID).

**A2 (2026-10-16, after the majority and pixel-logistic baselines, before any deep model).**
Endpoints, splits, metrics and the sealed test set are unchanged. Changes:

- **New baseline rung, `joint_prior`.** It predicts each joint's training erosion rate and nothing
  else. Joints differ in how often they erode, so knowing *which* joint is in the crop already
  gives above-chance AUROC. Every image model is reported against this floor, and so is the v2
  joint-identity embedding.
- **Pixel spacing (correcting A1).** A1 treated the different 0.15 and 0.175 mm/px spacings as a
  possible shortcut. The released images appear to be already resampled to 0.175 mm/px, as the
  paper states: HMCRD originals of 1670 px at 0.15 mm become 1431 px, which matches 0.175 mm.
  `make crop-stats` verifies this. It compares the median crop size of each joint across centers
  and flags any center that is more than 15% off.
- **Pre-specified v2 ladder.** Each rung is compared with the one before it by paired
  patient-bootstrap ΔAUROC (Section 7), using cross-validated out-of-fold predictions only.
  1. `densenet121`: v1, crops resized to 224 px.
  2. `densenet121_native`: native pixel scale on a fixed canvas.
  3. `densenet121_v2`: adds the ordinal head over SvdH grades, a joint-identity embedding, and
     mirroring left hands to right-hand orientation.
  4. `densenet121_v2_mt`: adds joint-space narrowing as an auxiliary task. JSN crops come only
     from each fold's training patients.
  The primary endpoint is still P(erosion) = P(grade ≥ 1). The ordinal head additionally reports
  grade QWK, so results can be compared with the RAM-H1200 benchmark. That comparison is like for
  like in kind only, not on the same split.

**A3 (2026-10-17, before any deep-model result).** Findings from `crop-stats`, the speed test and the
`joint_prior` baseline.

- **Pixel scale.** It is consistent across centers: median crop size per joint is within 0.88–1.12×
  of the cross-center median, which confirms A2. The spacing difference is therefore not a center
  shortcut.
- **Crop sizes.** Finger crops (MCP, PIP, IP) are a fixed 170 px. Carpal and wrist crops vary from
  73 px (trapezium) to 267 px (radius).
- **Effect on v1.** v1 resizes every crop to 224 px, so finger crops are slightly *enlarged*, not
  downsampled. The expected gain from native scale is therefore consistent physical scale across
  joint types, not recovered resolution. The research doc's "largest single gain" expectation for
  Methods 1 is downgraded accordingly.
- **Canvas.** Set to 224, the p95 of the longer crop side (197 px) rounded up; 0.5% of crops are
  shrunk. v2 therefore costs the same compute as v1.
- **The floor is the joint prior.** `joint_prior` scored AUROC 0.630 (0.610–0.650) and the pixel
  logistic regression 0.624 (0.605–0.643). The pixel model learned nothing beyond which joint it
  was looking at, so every deep model is compared against `joint_prior`.
- **Aborted run.** The Mac run of `densenet121_cv` was stopped after 4 epochs of fold 0 at 1–3 h per
  epoch, which is impractical. Its inner-val AUROC trajectory (0.73 → 0.81) was not used for any
  decision. Deep-model cross-validation moves to a GPU (docs/KAGGLE.md) with unchanged code, folds
  and seeds.

**A4 (2026-10-26). First deep-model results (5-fold out-of-fold, Kaggle T4, commit 46a76e2) and the
round-3 decisions they drive.**

| Run | Joint AUROC (95% CI) | Minimal vs none | Hand any-erosion | Grade QWK |
|---|---|---|---|---|
| joint_prior | 0.630 (0.610–0.650) | 0.656 | 0.486 | – |
| densenet121 (v1) | 0.753 (0.731–0.775) | 0.713 | 0.714 | – |
| densenet121_v2 | 0.760 (0.744–0.776) | 0.728 | 0.685 | 0.433 (0.367–0.487) |

- **v1 vs joint prior: ΔAUROC +0.123 (0.093–0.154).** The model sees erosions, not just which joint
  is in the crop.
- **v2 vs v1: ΔAUROC +0.007 (−0.009 to +0.024).** The gain is not established. Under Section 7 none
  of v2's components is claimed as an improvement. v2 becomes the base because it is non-inferior
  (lower bound −0.009) and adds grade output for the clinical interface. This decision rests on
  functionality and is recorded here.
- **The pre-registered predictions in the 2026-10-26 conversation (v1 ≈ 0.84) were too optimistic.**
  They extrapolated from a single fold's inner-val AUROC on the aborted Mac run. Across folds,
  inner-val peaks were 0.73–0.83, consistent with the out-of-fold results.
- **Performance by erosion grade (exploratory, not pre-specified).** v1 AUROC is 0.713 for grade 1,
  0.829 for grade 2 and 0.881 for grade 3 vs grade 0. 71% of positive joints are grade 1, so the
  primary AUROC is dominated by the subtlest lesions. Grade 1 is also the grade with the poorest
  inter-reader agreement in SvdH scoring.
- **Per-joint (v2).** The weakest joints are PIP-S 0.624, PIP-R 0.641, MCP-R 0.647, IP 0.676 and
  PIP-I 0.683. The strongest are R 0.843, L 0.835 and U 0.807. This is the opposite of the literature,
  where the wrist is weakest. Finger joints are where early RA erosions typically start.
- **Training curves.** Inner-val AUROC peaked at epochs 2–8 while the training loss fell to
  0.05–0.1, which is overfitting with 150 training patients per fold.
- **Hand-context stacking (exploratory, cross-fitted by fold).** Adding the other joints' and the
  contralateral joint's scores did not help: ΔAUROC −0.005 (−0.009 to −0.001). Method 5 (hand-level
  aggregator) is dropped.
- **Calibration.** Platt ECE is 0.007 for v2, adequate for displayed probabilities.
- **Centers.** Out-of-fold AUROC by center is HMCRD 0.753, SARC 0.765 and SCGH 0.771. No center is
  an outlier.

**Round 3 (pre-specified, compared by paired ΔAUROC on out-of-fold predictions):**
1. `densenet121_v3`, compared against v2. It adds regularisation for the overfitting: lr 3e-5,
   weight decay 1e-2, label smoothing 0.05, EMA of weights at 0.995, and a 25-epoch schedule with
   patience 6.
2. `densenet121_v3_384`, compared against v3. It adds a 384-px zoom of the uniform-scale canvas,
   aimed at the weak finger joints.

Deferred until round 3 reads out: ensembling and TTA (final model only), erosion-mask supervision,
and domain-specific pretraining (I-JEPA on unlabelled hand radiographs). The overfitting finding is
the motivation for the JEPA study.

**A5 (2026-10-28). Round-3 results (commit 98d69f0) and the pre-specified final-model candidate.**

| Run | Joint AUROC (95% CI) | Δ vs comparator | Grade ≥2 vs 0 | QWK argmax / cross-fitted decoding |
|---|---|---|---|---|
| densenet121_v3 | 0.766 (0.747–0.785) | +0.006 vs v2 (−0.008 to +0.019) | 0.860 | 0.453 / 0.486 (0.419–0.537) |
| densenet121_v3_384 | 0.761 (0.740–0.783) | −0.005 vs v3 (−0.017 to +0.007) | 0.858 | 0.443 / 0.507 (0.449–0.554) |

- **Regularisation.** It changed the training dynamics as intended: smooth curves, peaks at epochs
  5–14. Out-of-fold AUROC did not change significantly.
- **The 384-px zoom did not help finger joints** (PIP 0.62–0.72 either way). Label composition
  explains the finger weakness better than resolution does: 80% of eroded PIP/IP joints are grade 1,
  against 59% at the wrist and carpus.
- **Ensembling (exploratory, decided after seeing the members).** Equal-weight averages of
  out-of-fold predictions on identical folds:
  - v3 + v3_384: 0.779 (0.759–0.799), Δ vs v3 +0.013 (+0.008 to +0.019).
  - All four DenseNet runs: 0.799 (0.781–0.817), Δ vs v3 +0.033 (+0.025 to +0.041). Grade ≥2 vs 0:
    0.893. Grade 1 vs 0: 0.762. By center: HMCRD 0.789, SARC 0.810, SCGH 0.802.

  The models make partly independent errors, so averaging reduces the variance from the small
  training set.

**Pre-specified for round 4, before any of its results.**
1. Two new ensemble members with the v3 recipe on other architectures: `convnext_tiny_v3` and
   `effnetv2s_v3`.
2. **The final-model candidate is `ens6`: the equal-weight average of all six CV-evaluated members**
   (densenet121, densenet121_v2, densenet121_v3, densenet121_v3_384, convnext_tiny_v3, effnetv2s_v3).
   No member selection and no weight tuning.
3. Its out-of-fold AUROC is the internal-validation result. The 0.799 four-member number above stays
   labelled exploratory.
4. Next, and only after round 4:
   - Retrain each member on all dev patients.
   - Lock the operating threshold (90% sensitivity) and Platt calibration from ens6 out-of-fold
     predictions.
   - Run leave-one-center-out for one member (single-model LOCO, stated as such).
   - Evaluate on the sealed test set once.

**A6 (2026-10-28). Round-4 results (commit cfcdada) and the final-model lock.**

| Run | Joint AUROC (95% CI) | Δ (95% CI) | Grade ≥2 vs 0 | QWK, cross-fitted decoding |
|---|---|---|---|---|
| convnext_tiny_v3 | 0.786 (0.767–0.804) | +0.020 vs v3 (+0.012 to +0.029) | – | 0.509 (0.439–0.565) |
| effnetv2s_v3 | 0.758 (0.738–0.777) | −0.008 vs v3 (−0.017 to +0.001) | – | 0.452 (0.382–0.512) |
| **ens6 (pre-specified in A5)** | **0.804 (0.786–0.822)** | **+0.018 vs best single member (+0.013 to +0.023)** | **0.897** | **0.531 (0.465–0.585)** |

Further detail for ens6:
- **Grade 1 vs 0:** 0.767. **Hand-level any erosion:** 0.727 (0.664–0.784).
- **By center:** HMCRD 0.793, SARC 0.815, SCGH 0.807. **Per fold:** 0.789–0.818.
- **At 90% sensitivity:** specificity 0.479, PPV 0.317, NPV 0.947.
- **Calibration:** Platt ECE 0.010.

The recompute of ens6 from the member files matches the Kaggle output to 1e-16. ens6 is the internal-validation result for the final model.

**Final steps, pre-specified before the sealed test is opened:**
1. `scripts/lock_model.py` writes `results/final_lock.json` from ens6 out-of-fold predictions only:
   - the member list;
   - the threshold for 90% sensitivity;
   - Platt (a, b);
   - the grade cut-offs, fitted on the four members whose out-of-fold files carry grade
     probabilities.

   The file is committed and tagged `model-lock`.
2. Each member is retrained on all 190 dev patients (`--scheme full`, 15% inner validation for early
   stopping, same configs).
3. Single-member leave-one-center-out for `convnext_tiny_v3`, the best single member, reported as
   single-model internal-external validation.
4. `scripts/final_test.py` evaluates the locked ensemble once on the sealed split, which is the
   official RAM-H1200 test split. It refuses to run without a committed lock and refuses to run
   twice. Results are reported whatever they are, and tagged `v1.0-test`.

The benchmark comparison is grade QWK on the same official test split, against the published 0.4522.

**A7 (2026-10-29). Sealed-test and leave-one-center-out results (commit 8f8bec9, lock tag `model-lock`).
Reported as run; no analysis below changed any locked parameter.**

`scripts/final_test.py` ran once on the official RAM-H1200 test split (101 patients, 267 hands,
4,272 joints, erosion prevalence 27.0%) with the six full-data members, threshold, Platt and grade
cut-offs from `results/final_lock.json`. The Kaggle working tree was marked `-dirty`, most likely from the
downloaded data, logs and new result files; this was not checked at run time.

- **Primary endpoint, joint-level any erosion AUROC: 0.820 (0.801–0.834).** This is 0.016 above the
  internal CV estimate (0.804), within its interval. Recomputed from `test_predictions.csv`: 0.8201.
- **AUPRC** 0.630 (0.542–0.689). **Grade 1 vs 0:** 0.784 (0.756–0.800). **Grade ≥ 2 vs 0:** 0.880
  (0.858–0.900).
- **Locked threshold (0.191, set for 90% sensitivity on OOF):** sensitivity 0.844, specificity
  0.621, PPV 0.452, NPV 0.915. Sensitivity fell short of the 90% target by 5.6 points; the
  threshold did not transfer exactly.
- **Calibration:** ECE of the raw ensemble probability 0.023; ECE after the locked Platt map 0.081.
  The Platt map was fitted on fold-model OOF predictions and does not transfer to the full-data
  members, which are more confident. Platt scaling is not used for the next version until it is
  refitted on held-out data from the full-data members; this decision is made after seeing the
  test and is labelled post hoc.
- **Grade agreement, QWK with locked cut-offs:** 0.588 (0.478–0.678), against the published
  RAM-H1200 benchmark of 0.4522 on the same split. The lower confidence bound is above the
  benchmark. Balanced accuracy 0.523.
- **Hand-level any erosion:** 0.608 (0.431–0.776). Not interpretable: only 12 of 267 test hands are
  erosion-free, below the 30-hand rule in Section 3.
- **By center:** HMCRD 0.812 (95 of 101 patients). HU2 (1 patient), SARC (3) and SCGH (2) are too
  small to interpret, so the sealed test is effectively a single-center result.
- **By joint:** weakest IP 0.654, PIP-S 0.665, CMC-T 0.686; strongest R 0.924, L 0.989. Same pattern
  as CV.

**Leave-one-center-out, single model `convnext_tiny_v3` (CV 0.786):** pooled AUROC 0.697
(0.675–0.718); held-out HMCRD 0.706 (trained on 76 patients), SARC 0.760, SCGH 0.758. 352 joints
from centers with < 50 hands were train-only. Holding out a hospital costs 0.03–0.08 AUROC for a
single model, part of which is the smaller training set. The pooled value sits below every center
because scores are not calibrated across folds. This is the honest estimate of the generalisation
gap to a new site, and it is why external validation at a different institution comes before any
clinical claim.
