# Model card: Project Arthritis v2 erosion ensemble (v1.0)

> **Research use only. Not a medical device. Not for diagnosis or treatment decisions.**

## Model

| | |
|---|---|
| What it does | Scores each hand-radiograph joint crop for bone erosion (SvdH erosion ≥ 1) and estimates an erosion grade (0, 1, 2, 3, 5) |
| Architecture | Equal-weight ensemble of 6 ImageNet-pretrained CNNs: ConvNeXt-T v3, EfficientNetV2-S v3, DenseNet121 v1, v2, v3 and v3 at 384 px. Ordinal grade heads and a joint-identity embedding (v2 and later) |
| Input | Pre-cropped joint ROI at about 0.175 mm/px (RAM-H1200 format). No whole-hand input yet |
| Output | Erosion probability per joint (raw ensemble average), flag at threshold 0.191, grade estimate |
| Lock | `results/final_lock.json`, git tag `model-lock`; test commit 8f8bec9, tag `v1.0-test` |
| Licence | Code MIT. Weights CC BY-NC-SA 4.0 (derived from RAM-H1200): non-commercial |

## Intended use

- **Intended:** a second reader that helps a rheumatologist spot and track erosions on follow-up hand
  radiographs, in research settings.
- **Not intended:** standalone RA diagnosis (early RA often has normal radiographs), screening of
  people without an RA work-up, use without a clinician reviewing the image, or use on feet,
  non-PA views, children or non-RA arthropathies.
- **Future use under study:** triage support for non-specialist physicians. This needs separate
  validation with those users and is not supported by the results below.

## Data

RAM-H1200: 1,200 hand radiographs, 291 patients, 6 hospitals in Sapporo, Japan, with expert SvdH
scores for 16 joints per hand. Development: 190 patients (14,928 joints). Sealed test: the official
test split, 101 patients (4,272 joints), opened once.

## Performance

| Metric | Internal CV (190 pts) | Sealed test (101 pts) |
|---|---|---|
| Joint AUROC, any erosion | 0.804 (0.786–0.822) | **0.820 (0.801–0.834)** |
| Grade 1 vs 0 | 0.767 | 0.784 (0.756–0.800) |
| Grade ≥ 2 vs 0 | 0.897 | 0.880 (0.858–0.900) |
| Sensitivity / specificity at locked threshold | 0.90 / 0.479 | 0.844 / 0.621 |
| PPV / NPV at locked threshold | 0.317 / 0.947 | 0.452 / 0.915 |
| Grade QWK (benchmark 0.452) | 0.531 (0.465–0.585) | **0.588 (0.478–0.678)** |
| ECE (raw / locked Platt) | — / 0.010 | 0.023 / 0.081 |

Leave-one-center-out, single ConvNeXt-T: pooled 0.697 (0.675–0.718); per held-out center 0.71–0.76.
95% CIs from a patient-clustered bootstrap (2,000 resamples).

## Limitations

1. **One country, one city, one scoring group.** All data are from Sapporo. The sealed test is 94%
   one hospital (HMCRD). No external validation yet; LOCO suggests a 0.03–0.08 AUROC drop at a new site.
2. **Subtle erosions are the weak point.** Grade 1 is 71% of eroded joints and the grade readers
   disagree on most; finger IP/PIP joints score 0.65–0.74.
3. **Moderate specificity.** At the threshold that catches about 84% of eroded joints, about half of
   flagged joints are false positives. Output must be reviewed, not acted on directly.
4. **Calibration:** the raw probability is well calibrated on the test set (ECE 0.023); the locked
   Platt map is not (0.081) and is not used in the app.
5. **Hand-level and patient-level** performance is not established (12 erosion-free test hands).
6. **No joint detector yet:** inputs must be crops in the RAM-H1200 format. Whole-hand inference
   depends on a detector that has not been validated.
7. **No reader study:** benefit to clinicians is not measured.

## Ethical and regulatory notes

Not cleared by any regulator. Prospective use would need an ethics-approved study and, for clinical
use, regulatory review as software as a medical device.
