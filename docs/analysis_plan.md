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
