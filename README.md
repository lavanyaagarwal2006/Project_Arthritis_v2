# Project Arthritis v2: erosion detection on hand radiographs

> **Research use only. Not a medical device. Not for diagnosis.**

Deep-learning models that flag **bone erosions** on hand radiographs. Erosions are the radiographic
hallmark of erosive rheumatoid arthritis (RA). The intended use is a second reader for
rheumatologists monitoring joint damage between visits. It is not a standalone RA diagnosis: early
RA can have normal radiographs.

Supersedes [Project_Arthritis_v1](https://github.com/lavanyaagarwal2006/Project_Arthritis_v1).

## Headline result

On **RAM-H1200** (1,200 hand radiographs from 291 patients at 6 hospitals in Sapporo, Japan), the
pre-specified, locked 6-model ensemble reaches:

| Metric | Internal CV (190 patients, 14,928 joints) | **Sealed test** (101 patients, 4,272 joints, run once) |
|---|---|---|
| **Joint-level erosion AUROC** (any erosion, SvdH ≥ 1) | 0.804 (0.786–0.822) | **0.820 (0.801–0.834)** |
| Definite erosions (SvdH ≥ 2 vs 0) | 0.897 | 0.880 (0.858–0.900) |
| Minimal erosions (SvdH 1 vs 0) | 0.767 | 0.784 (0.756–0.800) |
| Erosion grade agreement, QWK (published benchmark 0.452) | 0.531 (0.465–0.585) | **0.588 (0.478–0.678)** |
| Sensitivity / specificity at the locked threshold | 0.90 / 0.479 | 0.844 / 0.621 |
| NPV at the locked threshold | 0.947 | 0.915 |
| Leave-one-hospital-out, single ConvNeXt-T | 0.697 pooled; 0.71–0.76 per hospital | — |

- 95% CIs come from a patient-clustered bootstrap (2,000 resamples).
- The ensemble and every analysis decision were pre-registered in
  [`docs/analysis_plan.md`](docs/analysis_plan.md) before the results they govern (amendments A1–A7).
- The model was locked in [`results/final_lock.json`](results/final_lock.json) (tag `model-lock`)
  before the official test split was opened once (tag `v1.0-test`).
- **Read the limits:** the test set is 94% one hospital, all data are from one city, and holding out
  a hospital costs 0.03–0.08 AUROC. See the [model card](docs/model_card.md).

### How the model got there

| Model | Joint AUROC (95% CI) | What changed |
|---|---|---|
| Majority class | 0.495 | chance |
| Joint prior (per-joint erosion rate only) | 0.630 (0.610–0.650) | the real floor: which joint it is |
| Pixel logistic regression | 0.624 (0.605–0.643) | no better than the joint prior |
| DenseNet121 (v1) | 0.753 (0.731–0.775) | +0.12 over the joint prior |
| DenseNet121 v2 (native scale, ordinal grades, joint identity) | 0.760 (0.744–0.776) | adds grade output |
| DenseNet121 v3 (regularisation, EMA) | 0.766 (0.747–0.785) | smoother training |
| ConvNeXt-T v3 | 0.786 (0.767–0.804) | best single model |
| **Equal-weight ensemble of 6 models** | **0.804 (0.786–0.822)** | +0.018 over the best single model |

The full table, including the runs that did not help, is in `results/summary.csv`.

### What we learned

- **Erosion size drives difficulty.** 71% of eroded joints are grade 1, the subtlest grade and the one
  readers disagree on most. AUROC rises from 0.77 (grade 1) to 0.90 (grade ≥ 2).
- **Finger joints are hardest** (PIP AUROC 0.66–0.73), because 80% of eroded PIP/IP joints are
  grade 1. Zooming finger crops did not help; ensembling did.
- **The model overfits within a few epochs** on about 150 training patients per fold. This motivates
  the planned self-supervised pretraining study.
- **No hospital is an outlier in CV** (AUROC by hospital 0.79–0.82), but a model that has never seen
  a hospital does worse there (LOCO 0.71–0.76). External validation comes before any clinical claim.
- **The locked Platt calibration did not transfer** to the full-data models (test ECE 0.081 vs 0.023
  for the raw probability), so the app uses the raw ensemble probability.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Pre-registered analysis plan with dated amendments (A1–A7) | done |
| 1 | Joint-level erosion detection: baselines → 6-model ensemble, 5-fold CV | done (0.804) |
| 1b | Leave-one-center-out validation and the sealed official test split | done (test 0.820) |
| 2 | Automatic joint detection for whole-hand uploads | next |
| 3 | Research-use web demo with per-joint heatmaps | next |
| 4 | I-JEPA self-supervised pretraining and a label-efficiency study | planned |
| 5 | External validation at a different institution or country | planned (data access) |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash scripts/download_data.sh        # gated: accept the terms on Hugging Face first (docs/data.md)
make manifest                        # folds are committed in splits/, so they are not regenerated
make test && make smoke2             # unit tests + full pipeline on synthetic data
```

Deep models were trained on free Kaggle T4 GPUs ([`docs/KAGGLE.md`](docs/KAGGLE.md)). Each run
writes `results/<run>/` containing its config, git SHA, out-of-fold predictions and metrics.

| Step | Command |
|---|---|
| Train one model, 5-fold CV | `python -m arthritis.train --exp convnext_tiny_v3 --run-name convnext_tiny_v3_cv` |
| Evaluate (CIs, calibration, per-joint, QWK) | `python -m arthritis.evaluate results/convnext_tiny_v3_cv` |
| Ensemble out-of-fold predictions | `python scripts/ensemble.py --name ens6_cv results/...` |
| Lock the final model (OOF only) | `python scripts/lock_model.py` |
| One-time sealed test | `python scripts/final_test.py` (refuses without a committed lock, refuses to run twice) |

## Layout

```
configs/       base.yaml + exp/<name>.yaml (one file per experiment)
docs/          analysis_plan.md (pre-registration + amendments), model_card.md, data.md, KAGGLE.md, WORKFLOW.md
scripts/       data download/inspection, crop stats, speed test, ensembling, model lock, final test
splits/        committed joint manifest and patient-grouped folds (no images, ever)
src/arthritis/ data, models, training, metrics, evaluation, inference
results/       per-run metrics, figures and out-of-fold predictions; summary.csv; final_lock.json
tests/         unit tests: metrics, splits, patient keys, ordinal grades
```

## Data and licences

- **Data:** [RAM-H1200](https://huggingface.co/datasets/TokyoTechMagicYang/RAM-H1200-v1) (Yang et al.,
  NeurIPS 2026 Datasets & Benchmarks, [arXiv:2605.05616](https://arxiv.org/abs/2605.05616)),
  CC BY-NC-SA 4.0. No images are redistributed here.
- **Trained weights** are derived from that data and are shared under the same non-commercial,
  share-alike terms.
- **Code:** MIT (see `LICENSE`).
