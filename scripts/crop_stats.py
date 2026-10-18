"""Crop-size report: picks the native-mode canvas and checks pixel scale is consistent across centers.

Reads only image headers (fast). Two questions:
  1. How big are the joint crops? -> canvas = 95th percentile of the longer side, rounded up to 32.
  2. Is 1 px the same physical size in every center? The same joint should have a similar median crop
     size in every center. A center >15% off suggests the images were NOT resampled to a common
     mm/px, so the model could tell centers apart by scale.

Usage: python scripts/crop_stats.py [--manifest splits/joints_manifest.csv] [--root data/RAM-H1200-v1]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="splits/joints_manifest.csv")
    ap.add_argument("--root", default="data/RAM-H1200-v1")
    ap.add_argument("--min-hands", type=int, default=50)
    args = ap.parse_args()
    root = Path(args.root)
    df = pd.read_csv(args.manifest, dtype={"patient_id": str})
    sizes = [Image.open(root / p).size for p in df.img_path]  # (w, h), header only
    df["w"], df["h"] = [s[0] for s in sizes], [s[1] for s in sizes]
    df["long"] = df[["w", "h"]].max(axis=1)

    print("crop size (longer side, px) by joint:")
    q = df.groupby("joint")["long"].describe(percentiles=[.5, .95])[["count", "min", "50%", "95%", "max"]]
    print(q.round(0).astype(int).to_string())
    p95 = df["long"].quantile(0.95)
    canvas = int(math.ceil(p95 / 32) * 32)
    print(f"\noverall: median {df['long'].median():.0f}, p95 {p95:.0f}, max {df['long'].max():.0f}")
    print(f"recommended canvas: {canvas}  (crops larger than this get shrunk: {(df['long'] > canvas).mean():.1%})")

    hands = df.groupby("center").case.nunique()
    big = hands[hands >= args.min_hands].index
    med = df[df.center.isin(big)].pivot_table(index="joint", columns="center", values="long", aggfunc="median")
    ratio = med.div(med.median(axis=1), axis=0)
    print("\nmedian crop size per center, relative to the joint's cross-center median (1.00 = same scale):")
    print(ratio.round(2).to_string())
    off = (ratio.sub(1).abs() > 0.15).mean()
    flag = [c for c, v in off.items() if v > 0.5]
    print("\nscale check:", "OK, consistent across centers" if not flag else
          f"WARNING: {flag} crops differ by >15% for most joints; images may not share one mm/px")


if __name__ == "__main__":
    main()
