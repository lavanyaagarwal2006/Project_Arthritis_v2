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
2. v1 4-block CNN, ported to PyTorch, trained from scratch
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

(none yet)
