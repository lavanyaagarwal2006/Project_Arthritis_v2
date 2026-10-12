# How we work in this repo

## Branches and commits
- `main` holds only work whose results are logged. Every change happens on a short branch:
  `exp/densenet-joint-crops`, `fix/manifest-bbox`, `docs/model-card`.
- Commit messages follow `type: summary`. Types: `feat`, `fix`, `exp`, `docs`, `data`, `refactor`.
  For example: `exp: densenet121 5-fold CV, joint AUROC 0.xx [0.xx, 0.xx]`.
- An experiment commit includes the config and `results/<run>/{metrics.json, figures/, run_meta.json}`.
  It never includes checkpoints or images.

## The research rules (non-negotiable)
1. `docs/analysis_plan.md` is frozen with tag `sap-v1.0` before the first real training run.
   Any change after that goes in its Amendments section, with a date and a reason.
2. The sealed test split is evaluated **once**, by `final_test` (to be written), after the model
   and threshold are locked. Tag that commit `v1.0-test`.
3. Every number on the resume or in a write-up comes from `results/summary.csv` at a tagged commit.
4. Report every run in `results/summary.csv`, including the bad ones.

## Tags
- `sap-v1.0`: analysis plan frozen
- `v0.1-baselines`: majority, pixel logistic regression, v1 CNN
- `v0.2-pretrained`: ResNet-50, DenseNet121, ConvNeXt-T
- `v0.3-loco`: leave-one-center-out
- `v1.0-test`: sealed test result
