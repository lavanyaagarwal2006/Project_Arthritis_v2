"""Average the out-of-fold predictions of several CV runs (same folds) and evaluate the result.

Members are averaged with equal weight: no weights are tuned on the out-of-fold data, so the only
selection is which runs are included, and the plan pre-specifies that (all CV-evaluated members).
Grade probabilities (gp*) are averaged over the members that have them.

Usage: python scripts/ensemble.py --name ens_all results/densenet121_cv results/densenet121_v2_cv ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from arthritis.config import git_sha  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    key = ["case", "joint"]
    frames = [pd.read_csv(r / "oof_predictions.csv", dtype={"patient_id": str}).sort_values(key).reset_index(drop=True)
              for r in args.runs]
    base = frames[0]
    for r, f in zip(args.runs, frames):
        if len(f) != len(base) or not (f[key].values == base[key].values).all() or not (f.fold.values == base.fold.values).all():
            sys.exit(f"{r} does not cover the same joints/folds as {args.runs[0]}")
    out = base.drop(columns=[c for c in base.columns if c.startswith("gp") or c.startswith("grade_pred")]).copy()
    out["prob"] = np.mean([f.prob.values for f in frames], axis=0)
    gp_frames = [f for f in frames if "gp0" in f.columns]
    if gp_frames:
        cols = sorted([c for c in gp_frames[0].columns if c.startswith("gp")], key=lambda c: int(c[2:]))
        P = np.mean([f[cols].values for f in gp_frames], axis=0)
        for i, c in enumerate(cols):
            out[c] = P[:, i]
        out["grade_pred_idx"] = P.argmax(1)
        if "grade_idx" not in out:
            from arthritis.config import load_config
            from arthritis.ordinal import level_index
            out["grade_idx"] = level_index(out["score"].values, load_config()["levels"]["BE"])
    run_dir = Path("results") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(run_dir / "oof_predictions.csv", index=False)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "exp_name": args.name, "scheme": "cv-ensemble", "git_sha": git_sha(),
        "members": [str(r) for r in args.runs], "grade_members": len(gp_frames)}, indent=2))
    print(f"wrote {run_dir}/oof_predictions.csv ({len(frames)} members, {len(gp_frames)} with grades); "
          f"next: python -m arthritis.evaluate {run_dir}")


if __name__ == "__main__":
    main()
