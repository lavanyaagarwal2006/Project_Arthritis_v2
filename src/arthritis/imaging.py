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


def fit_canvas(a: np.ndarray, canvas: int) -> tuple[np.ndarray, bool]:
    """Place a crop on a canvas x canvas square WITHOUT rescaling, so 1 px means the same distance for
    every joint and every center (RAM-H1200 images are resampled to 0.175 mm/px). A crop larger than the
    canvas is shrunk to fit (returned flag = True) — `make crop-stats` picks a canvas where this is rare."""
    h, w = a.shape
    shrunk = max(h, w) > canvas
    if shrunk:
        s = canvas / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        a = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BILINEAR), dtype=np.float32)
        h, w = a.shape
    out = np.zeros((canvas, canvas), dtype=np.float32)
    y0, x0 = (canvas - h) // 2, (canvas - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    return out, shrunk
