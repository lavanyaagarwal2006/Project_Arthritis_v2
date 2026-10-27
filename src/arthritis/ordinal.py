"""Ordinal (cumulative-threshold) grade handling. Pure numpy so it is testable without torch.

A grade on levels L = [l0 < l1 < ... < lK] is predicted as K threshold probabilities
p_k = P(grade > l_k), k = 0..K-1. Erosion present (grade >= 1) is p_0, so the binary endpoint and
its AUROC are unchanged; the other thresholds add supervision from the grade (a grade-3 erosion is a
'more confident positive' than a grade-1). A binary head is the special case K = 1.
"""
from __future__ import annotations

import numpy as np


def level_index(scores, levels) -> np.ndarray:
    """Map raw SvdH scores to 0..K by rank in `levels`; a value between levels maps to the level below."""
    lv = np.asarray(sorted(levels))
    s = np.asarray(scores)
    idx = np.searchsorted(lv, s, side="right") - 1
    return np.clip(idx, 0, len(lv) - 1)


def threshold_targets(idx, k: int) -> np.ndarray:
    """[N] grade indices -> [N, k] 0/1 targets for 'grade index > j'."""
    idx = np.asarray(idx)[:, None]
    return (idx > np.arange(k)[None, :]).astype(np.float32)


def class_probs(p_gt) -> np.ndarray:
    """[N, K] threshold probabilities -> [N, K+1] class probabilities (monotonicity enforced by cumulative min)."""
    p = np.minimum.accumulate(np.clip(np.asarray(p_gt, float), 0, 1), axis=1)
    n = p.shape[0]
    upper = np.concatenate([np.ones((n, 1)), p], axis=1)
    lower = np.concatenate([p, np.zeros((n, 1))], axis=1)
    return np.clip(upper - lower, 0, 1)


def decode_with_cutoffs(expected, cutoffs) -> np.ndarray:
    """Expected grade index -> class index using sorted cut-offs (len K)."""
    return np.searchsorted(np.sort(np.asarray(cutoffs)), np.asarray(expected), side="right")


def fit_cutoffs(expected, y, k_classes: int, score_fn, n_grid: int = 41, n_pass: int = 3) -> np.ndarray:
    """Coordinate search for cut-offs on the expected grade that maximise score_fn(y, pred).
    Corrects the upward grade shift caused by balanced sampling. Fit on training folds only."""
    e = np.asarray(expected, float)
    cuts = np.arange(1, k_classes) - 0.5
    grid = np.quantile(e, np.linspace(0.01, 0.99, n_grid))
    for _ in range(n_pass):
        for i in range(len(cuts)):
            best, best_c = -np.inf, cuts[i]
            for c in grid:
                trial = cuts.copy(); trial[i] = c
                sc = score_fn(y, decode_with_cutoffs(e, trial))
                if sc > best:
                    best, best_c = sc, c
            cuts[i] = best_c
    return np.sort(cuts)
