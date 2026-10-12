"""Image loading and preprocessing shared by training, evaluation and the web app (no torch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_roi(root: Path, img_path: str, bbox=None) -> np.ndarray:
    """Grayscale ROI in [0, 1]. Crops by 'x,y,w,h' bbox (+10% context) when given. Robust 1-99 percentile scaling."""
    img = Image.open(Path(root) / img_path).convert("L")
    if isinstance(bbox, str) and bbox:
        x, y, w, h = map(float, bbox.split(","))
        pad = 0.10 * max(w, h)
        img = img.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))
    a = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def letterbox(a: np.ndarray) -> np.ndarray:
    """Pad to square without distorting aspect ratio (erosions are shape features)."""
    h, w = a.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=a.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return out
