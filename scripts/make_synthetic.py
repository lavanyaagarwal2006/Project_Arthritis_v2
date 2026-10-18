"""Synthetic dataset in the RAM-H1200 benchmark layout, for pipeline tests only.

Each "joint" is a bright blob with a dark notch whose size grows with the SvdH score.
Patients have several hands (bilateral + follow-up), spread across 3 centers, so grouping
bugs show up as leakage. Numbers from this data mean nothing about real performance.

Usage: python scripts/make_synthetic.py data/synthetic [--patients 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

JOINTS = ["R", "U", "IP", "L", "MCP-T", "MCP-I", "MCP-M", "MCP-R", "MCP-S", "CMC-T",
          "PIP-I", "PIP-M", "PIP-R", "PIP-S", "S", "Tm"]
JSN_JOINTS = ["MCP-T", "MCP-I", "MCP-M", "MCP-R", "MCP-S", "PIP-I", "PIP-M", "PIP-R", "PIP-S",
              "CMC-M", "CMC-R", "CMC-S", "SC", "SR", "STT"]


def joint_image(rng, score: int, size: int = 72) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    r = size * rng.uniform(0.28, 0.34)
    img = 60 + 120 * (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2) + rng.normal(0, 18, (size, size))
    if score > 0:  # erosion: a dark notch on the blob edge
        ang = rng.uniform(0, 2 * np.pi)
        nx, ny = cx + r * np.cos(ang), cy + r * np.sin(ang)
        nr = 2.5 + 1.8 * score
        img[((xx - nx) ** 2 + (yy - ny) ** 2) < nr ** 2] = 60
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    """Mirrors the real RAM-H1200 layout: SvdH_Scoring/SvdH_BE_Scoring/<split>/<case>/<JOINT>.bmp,
    case names JP_<CENTER>_P<num>_<date>_<id>_<L|R>, a per-study Metadata.xlsx whose patient numbers
    restart at 1 in every center (so the same number means different people in different centers)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    base = args.out / "SvdH_Scoring" / "SvdH_BE_Scoring"
    jbase = args.out / "SvdH_Scoring" / "SvdH_JSN_Scoring"
    meta = []
    splits = {"train": {}, "val": {}, "test": {}}
    jsplits = {"train": {}, "val": {}, "test": {}}
    centers = ["HMCRD", "SARC", "SCGH", "HU2"]
    per_center_counter = {c: 0 for c in centers}
    for pi in range(args.patients):
        center = centers[0] if pi % 10 < 4 else centers[1] if pi % 10 < 7 else centers[2] if pi % 10 < 9 else centers[3]
        per_center_counter[center] += 1
        num = per_center_counter[center]
        split = "test" if pi % 5 == 0 else ("val" if pi % 7 == 0 else "train")
        is_ra = int(rng.random() < 0.85)
        severity = rng.uniform(0, 0.35) if is_ra else 0.0
        for v in range(int(rng.integers(1, 4))):
            stem = f"JP_{center}_P{num:04d}_{2012 + 3 * v}0{v + 1}15_{rng.integers(1000, 9999)}"
            for side in ("L", "R"):
                case = f"{stem}_{side}"
                d = base / split / case
                d.mkdir(parents=True, exist_ok=True)
                scores = {}
                for j in JOINTS:
                    s = int(rng.choice([1, 2, 3, 5], p=[0.6, 0.25, 0.05, 0.1])) if rng.random() < severity * (1 + 0.3 * v) else 0
                    scores[f"BE_{j}"] = s
                    Image.fromarray(joint_image(rng, min(s, 3))).save(d / f"{j}.bmp")
                splits[split][f"{case}.bmp"] = scores
                jd = jbase / split / case
                jd.mkdir(parents=True, exist_ok=True)
                jscores = {}
                for j in JSN_JOINTS:
                    s = int(rng.choice([1, 2, 3, 4])) if rng.random() < severity else 0
                    jscores[f"JSN_{j}"] = s
                    Image.fromarray(joint_image(rng, min(s, 3), size=60)).save(jd / f"{j}.bmp")
                jsplits[split][f"{case}.bmp"] = jscores
            meta.append({"Mapped Image Stem": stem, "StudyID": v + 1, "Normalized PatientID": num, "isRA": is_ra,
                         "Sex": "F", "Age": 55.0, "Center": center, "PixelSpacing": "[0.175, 0.175]",
                         "ImageSize": "[1722, 1430]", "LR": "LR"})
    for s, rec in splits.items():
        (base / s).mkdir(parents=True, exist_ok=True)
        (base / s / "_annotation_be_scores.json").write_text(json.dumps(rec))
        (jbase / s).mkdir(parents=True, exist_ok=True)
        (jbase / s / "_annotation_jsn_scores.json").write_text(json.dumps(jsplits[s]))
    pd.DataFrame(meta).to_excel(args.out / "Metadata.xlsx", index=False)
    print(f"synthetic dataset: {args.patients} patients, {len(meta)} studies -> {args.out}")


if __name__ == "__main__":
    main()
