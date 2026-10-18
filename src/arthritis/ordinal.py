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
