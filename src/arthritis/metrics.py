"""Metrics with patient-clustered bootstrap CIs. Pure numpy/scipy/sklearn (no torch).

Every function takes y (0/1), p (probability) and, where relevant, groups (patient IDs).
Joints from one patient are correlated, so CIs resample *patients*, not joints.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def sens_at_spec(y, p, spec: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y, p, sens: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= sens - 1e-12
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def threshold_for_sensitivity(y, p, target: float = 0.90) -> float:
    """Highest threshold that still reaches the target sensitivity (fit on validation/OOF only)."""
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos)))  # need at least k positives at or above the threshold
    return float(pos[k - 1])


def confusion_at(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn)}


def ece(y, p, n_bins: int = 15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_curve(y, p, n_bins: int = 15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf.append(p[m].mean()); acc.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_temperature(y, p) -> float:
    """Temperature T minimizing NLL of sigmoid(logit/T). Fit on OOF/validation only."""
    y, z = np.asarray(y, float), _logit(p)

    def nll(log_t):
        q = 1 / (1 + np.exp(-z / np.exp(log_t)))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(p, t: float):
    return 1 / (1 + np.exp(-_logit(p) / t))


def fit_platt(y, p) -> tuple[float, float]:
    """Logistic recalibration sigmoid(a*logit + b). Unlike temperature scaling, the bias term
    corrects the prevalence shift caused by balanced sampling or class weights during training."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, ab: tuple[float, float]):
    a, b = ab
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def aggregate_to_hand(cases, groups, y, p, how: str = "max"):
    """Joint -> hand. A hand is positive if any joint is positive; its score is max (or noisy-OR) of joint probs."""
    import pandas as pd
    df = pd.DataFrame({"case": cases, "g": groups, "y": y, "p": p})
    if how == "max":
        agg = df.groupby("case").agg(g=("g", "first"), y=("y", "max"), p=("p", "max"))
    elif how == "noisy_or":
        agg = df.groupby("case").agg(g=("g", "first"), y=("y", "max"),
                                     p=("p", lambda s: 1 - np.prod(1 - np.clip(s, 0, 1 - 1e-9))))
    else:
        raise ValueError(how)
    return agg.index.values, agg.g.values, agg.y.values.astype(int), agg.p.values


METRICS = {
    "auroc": auroc,
    "auprc": auprc,
    "sens_at_90spec": lambda y, p: sens_at_spec(y, p, 0.90),
    "spec_at_90sens": lambda y, p: spec_at_sens(y, p, 0.90),
}


def cluster_bootstrap(y, p, groups, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Point estimate + percentile CI, resampling clusters (patients) with replacement."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    point = fn(y, p)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        if 0 < y[idx].sum() < len(idx):
            stats.append(fn(y[idx], p[idx]))
    stats = np.array([s for s in stats if np.isfinite(s)])
    if len(stats) < 0.9 * n_boot:
        print(f"note: only {len(stats)}/{n_boot} bootstrap draws were usable")
    lo, hi = (np.quantile(stats, [alpha / 2, 1 - alpha / 2]) if len(stats) else (np.nan, np.nan))
    return {"value": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(len(stats))}


def paired_auroc_difference(y, p_a, p_b, groups, n_boot: int = 2000, seed: int = 0):
    """AUROC(a) - AUROC(b) with a paired patient-clustered bootstrap CI."""
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    stacked = np.stack([p_a, p_b], 1)
    fn = lambda yy, pp: auroc(yy, pp[:, 0]) - auroc(yy, pp[:, 1])  # noqa: E731
    return cluster_bootstrap(y, stacked, groups, fn, n_boot, seed)


def full_report(y, p, groups, thr: float | None = None, n_boot: int = 2000, seed: int = 0, ece_bins: int = 15) -> dict:
    out = {"n": int(len(y)), "n_pos": int(np.sum(y)), "n_patients": int(len(np.unique(groups)))}
    for name, fn in METRICS.items():
        out[name] = cluster_bootstrap(y, p, groups, fn, n_boot, seed)
    out["ece"] = ece(y, p, ece_bins)
    if thr is not None:
        out["at_threshold"] = confusion_at(y, p, thr)
    return out
