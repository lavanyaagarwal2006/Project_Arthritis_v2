"""Evaluate a run's out-of-fold predictions. Pure numpy/sklearn, no torch.

Input:  results/<run>/oof_predictions.csv with columns
        case, joint, patient_id, center, fold, score, label, prob
Output: results/<run>/metrics.json, figures/, and one row appended to results/summary.csv

Usage:
  python -m arthritis.evaluate results/<run>
  python -m arthritis.evaluate results/<run> --compare results/<other_run>   # paired AUROC difference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .config import resolve


def cross_fitted(df: pd.DataFrame, method: str = "platt") -> tuple[np.ndarray, list]:
    """Fit the recalibrator on the other folds and apply it to each fold, so calibration is never scored on its own fit data."""
    out = np.empty(len(df)); params = []
    for k in sorted(df.fold.unique()):
        tr, te = (df.fold != k) if df.fold.nunique() > 1 else (df.fold == df.fold), df.fold == k
        if method == "temperature":
            t = M.fit_temperature(df.label[tr], df.prob[tr]); out[te.values] = M.apply_temperature(df.prob[te], t)
        else:
            t = M.fit_platt(df.label[tr], df.prob[tr]); out[te.values] = M.apply_platt(df.prob[te], t)
        params.append(t)
    return out, params


def evaluate(df: pd.DataFrame, n_boot: int = 2000, target_sens: float = 0.90, ece_bins: int = 15, seed: int = 0) -> dict:
    y, p, g = df.label.values, df.prob.values, df.patient_id.values
    thr = M.threshold_for_sensitivity(y, p, target_sens)
    res: dict = {"cohort": {"patients": int(df.patient_id.nunique()), "hands": int(df.case.nunique()),
                            "joints": int(len(df)), "joint_prevalence": float(y.mean())}}

    # Primary: joint level
    res["joint"] = M.full_report(y, p, g, thr, n_boot, seed, ece_bins)
    res["joint"]["threshold_note"] = (f"threshold chosen on these same OOF predictions for {target_sens:.0%} "
                                      "sensitivity; the operating point is optimistic until it is applied to the sealed test set")

    # Early detection: minimal erosion (BE=1) vs none (BE=0)
    sub = df[df.score <= 1]
    if sub.label.nunique() == 2:
        res["joint_minimal_erosion_vs_none"] = M.full_report(sub.label.values, sub.prob.values,
                                                             sub.patient_id.values, thr, n_boot, seed, ece_bins)

    # Key secondary: hand level (any erosion), max aggregation
    cases, hg, hy, hp = M.aggregate_to_hand(df.case.values, g, y, p, "max")
    hthr = M.threshold_for_sensitivity(hy, hp, target_sens)
    res["hand_any_erosion"] = M.full_report(hy, hp, hg, hthr, n_boot, seed, ece_bins) if 0 < hy.sum() < len(hy) \
        else {"note": "all hands have the same label; hand-level AUROC undefined"}

    # Per-fold spread (how stable is it?)
    per = {}
    for k, d in df.groupby("fold"):
        per[str(k)] = M.auroc(d.label, d.prob)
    vals = np.array([v for v in per.values() if np.isfinite(v)])
    res["per_fold_auroc"] = {"folds": per, "mean": float(vals.mean()) if len(vals) else None,
                             "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None}

    # Per-center (meaningful for leave-one-center-out runs)
    res["per_center_auroc"] = {str(c): M.auroc(d.label, d.prob) for c, d in df.groupby("center")}

    # Calibration
    p_t, temps = cross_fitted(df, "temperature")
    p_pl, platts = cross_fitted(df, "platt")
    res["calibration"] = {"ece_raw": M.ece(y, p, ece_bins),
                          "ece_temperature_scaled": M.ece(y, p_t, ece_bins),
                          "ece_platt": M.ece(y, p_pl, ece_bins),
                          "note": "both recalibrators are cross-fitted across folds; Platt also corrects the prevalence shift from balanced sampling",
                          "temperature_all_oof": M.fit_temperature(y, p), "platt_all_oof": M.fit_platt(y, p),
                          "per_fold_temperature": temps, "per_fold_platt": platts}
    return res


def make_figures(df: pd.DataFrame, res: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    fpr, tpr, _ = roc_curve(df.label, df.prob)
    a = res["joint"]["auroc"]
    ax[0].plot(fpr, tpr, lw=2, label=f"joint: AUROC {a['value']:.3f} ({a['ci_low']:.3f}–{a['ci_high']:.3f})")
    _, _, hy, hp = M.aggregate_to_hand(df.case.values, df.patient_id.values, df.label.values, df.prob.values)
    if 0 < hy.sum() < len(hy):
        f2, t2, _ = roc_curve(hy, hp)
        h = res["hand_any_erosion"]["auroc"]
        ax[0].plot(f2, t2, lw=2, ls="--", label=f"hand: AUROC {h['value']:.3f} ({h['ci_low']:.3f}–{h['ci_high']:.3f})")
    ax[0].plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="chance")
    ax[0].set(xlabel="1 − specificity", ylabel="sensitivity", title="ROC (out-of-fold, 95% patient-bootstrap CI)")
    ax[0].legend(loc="lower right", fontsize=8)

    conf, acc, _ = M.reliability_curve(df.label, df.prob)
    p_cal, _ = cross_fitted(df, "platt")
    conf2, acc2, _ = M.reliability_curve(df.label, p_cal)
    c = res["calibration"]
    ax[1].plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
    ax[1].plot(conf, acc, "o-", label=f"raw (ECE {c['ece_raw']:.3f})")
    ax[1].plot(conf2, acc2, "s-", label=f"Platt, cross-fitted (ECE {c['ece_platt']:.3f})")
    ax[1].set(xlabel="predicted probability", ylabel="observed erosion rate", title="Reliability (joint level)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc_calibration.png", dpi=150)
    plt.close(fig)


def append_summary(run_dir: Path, res: dict, meta: dict) -> None:
    j, h = res["joint"], res.get("hand_any_erosion", {})
    fmt = lambda d: f"{d['value']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"  # noqa: E731
    row = {"run": run_dir.name, "exp": meta.get("exp_name", ""), "scheme": meta.get("scheme", ""),
           "git_sha": meta.get("git_sha", ""),
           "joint_auroc": fmt(j["auroc"]), "joint_auprc": fmt(j["auprc"]),
           "joint_sens@90spec": fmt(j["sens_at_90spec"]),
           "minimal_vs_none_auroc": fmt(res["joint_minimal_erosion_vs_none"]["auroc"]) if "joint_minimal_erosion_vs_none" in res else "",
           "hand_auroc": fmt(h["auroc"]) if "auroc" in h else "",
           "ece_raw": f"{res['calibration']['ece_raw']:.3f}",
           "ece_temp": f"{res['calibration']['ece_temperature_scaled']:.3f}",
           "ece_platt": f"{res['calibration']['ece_platt']:.3f}"}
    path = resolve("results/summary.csv")
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df[df.get("run", pd.Series(dtype=str)) != run_dir.name] if len(df) else df,
                    pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    run = resolve(args.run_dir)
    df = pd.read_csv(run / "oof_predictions.csv", dtype={"patient_id": str})
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}

    res = evaluate(df, n_boot=args.n_boot)
    if args.compare:
        other = pd.read_csv(resolve(args.compare) / "oof_predictions.csv", dtype={"patient_id": str})
        mrg = df.merge(other[["case", "joint", "prob"]], on=["case", "joint"], suffixes=("", "_b"))
        assert len(mrg) == len(df), "runs must cover the same joints to compare"
        res["paired_auroc_diff_vs"] = {"other": args.compare.name,
                                       **M.paired_auroc_difference(mrg.label, mrg.prob, mrg.prob_b, mrg.patient_id, args.n_boot)}
    (run / "metrics.json").write_text(json.dumps(res, indent=2))
    make_figures(df, res, run / "figures")
    append_summary(run, res, meta)

    j = res["joint"]
    print(f"[{run.name}] joint AUROC {j['auroc']['value']:.3f} "
          f"(95% CI {j['auroc']['ci_low']:.3f}–{j['auroc']['ci_high']:.3f}), "
          f"AUPRC {j['auprc']['value']:.3f}, sens@90spec {j['sens_at_90spec']['value']:.3f}, "
          f"ECE raw {res['calibration']['ece_raw']:.3f} / temp {res['calibration']['ece_temperature_scaled']:.3f} / Platt {res['calibration']['ece_platt']:.3f}")
    if "joint_minimal_erosion_vs_none" in res:
        e = res["joint_minimal_erosion_vs_none"]["auroc"]
        print(f"   minimal erosion (BE=1) vs none: AUROC {e['value']:.3f} ({e['ci_low']:.3f}–{e['ci_high']:.3f})")
    if "auroc" in res.get("hand_any_erosion", {}):
        h = res["hand_any_erosion"]["auroc"]
        print(f"   hand-level any erosion: AUROC {h['value']:.3f} ({h['ci_low']:.3f}–{h['ci_high']:.3f})")
    if "paired_auroc_diff_vs" in res:
        d = res["paired_auroc_diff_vs"]
        print(f"   ΔAUROC vs {d['other']}: {d['value']:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f})")


if __name__ == "__main__":
    main()
