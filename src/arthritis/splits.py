"""Patient-grouped CV folds and leave-one-center-out (LOCO) assignment.

Output: splits/folds.csv with one row per patient:
  patient_id, center, role (dev | sealed), fold (0..k-1 for dev, -1 for sealed)

Guarantees, checked by assertions:
  - No patient appears in more than one fold.
  - No sealed-test patient appears in the dev pool.

Usage: python -m arthritis.splits
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import load_config, resolve


def make_folds(manifest: pd.DataFrame, n_folds: int, seed: int, sealed_split: str) -> pd.DataFrame:
    sealed_patients = set(manifest.loc[manifest.split == sealed_split, "patient_id"])
    dev = manifest[~manifest.patient_id.isin(sealed_patients)]
    dropped = manifest[(manifest.split != sealed_split) & manifest.patient_id.isin(sealed_patients)]
    if len(dropped):
        print(f"WARNING: {dropped.patient_id.nunique()} patients appear in the sealed split AND in train/val. "
              "Their train/val images are moved to sealed to prevent leakage.")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, idx) in enumerate(sgkf.split(dev, dev["label"], groups=dev["patient_id"])):
        for p in dev.iloc[idx]["patient_id"].unique():
            assert p not in fold_of, f"patient {p} assigned twice"
            fold_of[p] = k

    pts = manifest.drop_duplicates("patient_id")[["patient_id", "center"]].copy()
    pts["role"] = np.where(pts.patient_id.isin(sealed_patients), "sealed", "dev")
    pts["fold"] = pts.patient_id.map(fold_of).fillna(-1).astype(int)
    assert (pts.loc[pts.role == "sealed", "fold"] == -1).all()
    assert (pts.loc[pts.role == "dev", "fold"] >= 0).all()
    return pts.sort_values(["role", "fold", "patient_id"]).reset_index(drop=True)


def attach(manifest: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    m = manifest.drop(columns=[c for c in ("role", "fold") if c in manifest]).merge(
        folds[["patient_id", "role", "fold"]], on="patient_id", how="left")
    assert m["role"].notna().all()
    return m


def describe(m: pd.DataFrame) -> str:
    lines = ["role/fold | patients | hands | joints | joint prevalence"]
    for (role, fold), d in m.groupby(["role", "fold"]):
        lines.append(f"{role}/{fold} | {d.patient_id.nunique()} | {d.case.nunique()} | {len(d)} | {d.label.mean():.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()
    cfg = load_config(args.exp)
    man = pd.read_csv(resolve(cfg["data"]["manifest"]), dtype={"patient_id": str})
    folds = make_folds(man, cfg["cv"]["n_folds"], cfg["seed"], cfg["data"]["sealed_split"])
    out = resolve(cfg["data"]["folds"])
    folds.to_csv(out, index=False)
    print(describe(attach(man, folds)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
