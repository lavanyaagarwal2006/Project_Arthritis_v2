"""ONE-TIME evaluation of the locked ensemble on the sealed (official RAM-H1200 test) split.

Refuses to run unless results/final_lock.json is committed to git, and refuses to run twice.
Uses only what the lock file fixes: members, threshold, Platt parameters, grade cut-offs.

Usage (on a GPU machine, after the `--scheme full` member fits):
  PYTHONPATH=src python scripts/final_test.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from arthritis import metrics as M  # noqa: E402
from arthritis.config import git_sha, load_config, resolve  # noqa: E402
from arthritis.ordinal import class_probs, decode_with_cutoffs, level_index  # noqa: E402
from arthritis.splits import attach  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="results/final_lock.json")
    ap.add_argument("--out", default="results/final_test")
    args = ap.parse_args()
    lock_path, out = resolve(args.lock), resolve(args.out)
    tracked = subprocess.call(["git", "ls-files", "--error-unmatch", str(lock_path)], cwd=resolve("."),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if not tracked:
        sys.exit("final_lock.json is not committed. Lock the model first (scripts/lock_model.py, commit, tag).")
    if (out / "metrics.json").exists():
        sys.exit(f"{out} already exists: the sealed test has been evaluated once. It is not re-run.")
    lock = json.loads(lock_path.read_text())
    cfg = load_config()
    root = resolve(cfg["data"]["root"])
    man = attach(pd.read_csv(resolve(cfg["data"]["manifest"]), dtype={"patient_id": str}),
                 pd.read_csv(resolve(cfg["data"]["folds"]), dtype={"patient_id": str}))
    test = man[man.role == "sealed"].reset_index(drop=True)
    print(f"sealed test: {test.patient_id.nunique()} patients, {test.case.nunique()} hands, {len(test)} joints")

    from arthritis.infer import load_member, predict_member
    probs, grade_ps = [], []
    for exp in lock["members"]:
        meta = json.loads((resolve("results") / f"{exp}_full" / "model.json").read_text())
        member = load_member(exp, resolve(meta["checkpoint"]))
        p = predict_member(member, test, root)
        probs.append(p[:, 0])
        if p.shape[1] > 1 and exp in lock.get("grade_members", lock["members"]):
            grade_ps.append(class_probs(p))
        print(f"  {exp}: AUROC {M.auroc(test.label, p[:, 0]):.3f}")
    prob = np.mean(probs, axis=0)
    g = test.patient_id.values
    res = {"lock": lock, "git_sha": git_sha(),
           "cohort": {"patients": int(test.patient_id.nunique()), "hands": int(test.case.nunique()),
                      "joints": int(len(test)), "joint_prevalence": float(test.label.mean())}}
    res["joint"] = M.full_report(test.label.values, prob, g, lock["threshold"])
    s1, s2 = test.score <= 1, (test.score == 0) | (test.score >= 2)
    res["minimal_vs_none"] = M.cluster_bootstrap(test.label[s1].values, prob[s1.values], g[s1.values], M.auroc)
    res["grade2plus_vs_none"] = M.cluster_bootstrap(test.label[s2].values, prob[s2.values], g[s2.values], M.auroc)
    cases, hg, hy, hp = M.aggregate_to_hand(test.case.values, g, test.label.values, prob)
    res["hand_any_erosion"] = M.cluster_bootstrap(hy, hp, hg, M.auroc)
    res["calibration"] = {"ece_raw": M.ece(test.label, prob), "ece_platt_locked": M.ece(test.label, M.apply_platt(prob, lock["platt"]))}
    res["per_center_auroc"] = {str(c): M.auroc(d.label, prob[d.index]) for c, d in test.groupby("center")}
    res["per_joint_auroc"] = {str(j): M.auroc(d.label, prob[d.index]) for j, d in test.groupby("joint")}
    if grade_ps and "grade_cutoffs" in lock:
        from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
        P = np.mean(grade_ps, axis=0)
        gi = level_index(test.score.values, cfg["levels"]["BE"])
        pred = decode_with_cutoffs(P @ np.arange(P.shape[1]), lock["grade_cutoffs"])
        qwk = lambda a, b: cohen_kappa_score(a, b, weights="quadratic")  # noqa: E731
        res["grade"] = {"qwk_locked_decoding": M.cluster_bootstrap(gi, pred, g, qwk),
                        "balanced_accuracy": float(balanced_accuracy_score(gi, pred)),
                        "benchmark": "RAM-H1200 paper, official test: QWK 0.4522 (MedMamba), BACC 35.87% (ResNet)"}
    out.mkdir(parents=True, exist_ok=True)
    test.assign(prob=prob).to_csv(out / "test_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    j, f = res["joint"], lambda d: f"{d['value']:.3f} ({d['ci_low']:.3f}–{d['ci_high']:.3f})"
    print(f"\nSEALED TEST  joint AUROC {f(j['auroc'])} | grade1 vs none {f(res['minimal_vs_none'])} | "
          f"grade>=2 vs none {f(res['grade2plus_vs_none'])} | hand {f(res['hand_any_erosion'])}")
    at = j["at_threshold"]
    print(f"  locked threshold {lock['threshold']:.3f}: sensitivity {at['sensitivity']:.3f}, specificity {at['specificity']:.3f}, "
          f"PPV {at['ppv']:.3f}, NPV {at['npv']:.3f}")
    if "grade" in res:
        q = res["grade"]["qwk_locked_decoding"]
        print(f"  grade QWK {f(q)}, balanced acc {res['grade']['balanced_accuracy']:.1%}  [benchmark: QWK 0.452, BACC 35.9%]")
    print(f"  per center: { {k: round(v, 3) for k, v in res['per_center_auroc'].items()} }")
    print(f"wrote {out}/metrics.json. Commit it and tag: git tag v1.0-test")


if __name__ == "__main__":
    main()
