"""Lock everything the sealed test will use, from out-of-fold predictions only, BEFORE the test is opened.

Writes results/final_lock.json:
  members           : experiment names of the pre-specified ensemble (A5)
  threshold         : ensemble P(erosion) giving >= 90% sensitivity on the OOF predictions
  platt             : (a, b) logistic recalibration fitted on the OOF predictions
  grade_cutoffs     : cut-offs on the expected grade, fitted on the OOF predictions of ordinal members
Commit and tag this file (`git tag model-lock`) before running scripts/final_test.py.

Usage: python scripts/lock_model.py [--oof results/ens6_cv/oof_predictions.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from arthritis import metrics as M  # noqa: E402
from arthritis.config import git_sha  # noqa: E402
from arthritis.ordinal import fit_cutoffs  # noqa: E402

MEMBERS = ["densenet121", "densenet121_v2", "densenet121_v3", "densenet121_v3_384", "convnext_tiny_v3", "effnetv2s_v3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="results/ens6_cv/oof_predictions.csv")
    ap.add_argument("--out", default="results/final_lock.json")
    args = ap.parse_args()
    if Path(args.out).exists():
        sys.exit(f"{args.out} already exists: the model is locked. Delete it only if the sealed test has NOT been run.")
    from sklearn.metrics import cohen_kappa_score
    d = pd.read_csv(args.oof)
    thr = M.threshold_for_sensitivity(d.label.values, d.prob.values, 0.90)
    a, b = M.fit_platt(d.label.values, d.prob.values)
    lock = {"members": MEMBERS, "threshold": thr, "target_sensitivity": 0.90, "platt": [a, b],
            "oof_source": args.oof, "oof_auroc": M.auroc(d.label, d.prob), "git_sha_at_lock": git_sha()}
    gp = sorted([c for c in d.columns if c.startswith("gp") and c[2:].isdigit()], key=lambda c: int(c[2:]))
    run_meta = Path(args.oof).parent / "run_meta.json"
    if run_meta.exists():  # members whose OOF carried grade probabilities = the grade sub-ensemble the cut-offs fit
        mem = json.loads(run_meta.read_text())["members"]
        lock["grade_members"] = [Path(m).name.removesuffix("_cv") for m in mem
                                 if "gp0" in open(Path(m) / "oof_predictions.csv").readline().split(",")]
    if gp:
        P = d[gp].values
        cuts = fit_cutoffs(P @ np.arange(P.shape[1]), d.grade_idx.values, P.shape[1],
                           lambda y, p: cohen_kappa_score(y, p, weights="quadratic"))
        lock["grade_cutoffs"] = [float(c) for c in cuts]
    Path(args.out).write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))
    print(f"\nwrote {args.out}. Now: git add {args.out} && git commit -m 'lock: final model' && git tag model-lock && git push --tags")


if __name__ == "__main__":
    main()
