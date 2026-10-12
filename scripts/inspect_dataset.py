"""Print the layout of a downloaded dataset so the manifest builder can be configured.

Reads only the folder structure, JSON keys, and spreadsheet headers and label counts.
It never opens model outputs, so it is safe to run before the analysis plan is frozen.

Usage: python scripts/inspect_dataset.py data/RAM-H1200-v1 [--out results/data_inspection.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def tree_summary(root: Path, max_depth: int = 4) -> list[str]:
    lines = []
    by_dir: dict[Path, Counter] = defaultdict(Counter)
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.parent.relative_to(root)
            if len(rel.parts) <= max_depth:
                by_dir[rel][p.suffix.lower() or "<none>"] += 1
            else:
                by_dir[Path(*rel.parts[:max_depth]) / "..."][p.suffix.lower() or "<none>"] += 1
    # collapse many sibling case folders into one line
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for d, c in by_dir.items():
        groups[(d.parent, tuple(sorted(c.items())))].append(d)
    for (parent, counts), dirs in sorted(groups.items(), key=lambda x: str(x[0][0])):
        label = str(dirs[0]) if len(dirs) == 1 else f"{parent}/<{len(dirs)} dirs, e.g. {dirs[0].name}>"
        lines.append(f"  {label}: " + ", ".join(f"{k}×{v}" for k, v in counts))
    return lines


def describe_json(p: Path) -> list[str]:
    try:
        obj = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return [f"  could not parse: {e}"]
    out = [f"  type={type(obj).__name__} len={len(obj) if hasattr(obj, '__len__') else '-'}"]
    if isinstance(obj, dict):
        keys = list(obj)[:3]
        out.append(f"  first keys: {keys}")
        if {"images", "annotations", "categories"} <= set(obj):
            cats = [c.get("name") for c in obj["categories"]]
            out.append(f"  COCO: {len(obj['images'])} images, {len(obj['annotations'])} annots, categories={cats}")
            out.append(f"  first image entry: {obj['images'][0]}")
        else:
            v = obj[keys[0]] if keys else None
            out.append(f"  first value: {json.dumps(v)[:400]}")
            if isinstance(v, dict):
                dist: dict[str, Counter] = defaultdict(Counter)
                for rec in obj.values():
                    if isinstance(rec, dict):
                        for k, s in rec.items():
                            if isinstance(s, (int, float)):
                                dist[k][int(s)] += 1
                if dist:
                    total = Counter()
                    for c in dist.values():
                        total.update(c)
                    out.append(f"  score distribution (all keys pooled): {dict(sorted(total.items()))}")
                    pos_hands = sum(1 for rec in obj.values() if isinstance(rec, dict) and any(
                        isinstance(s, (int, float)) and s >= 1 for s in rec.values()))
                    out.append(f"  hands with any score >= 1: {pos_hands}/{len(obj)}")
    return out


def describe_table(p: Path) -> list[str]:
    import pandas as pd
    try:
        df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        return [f"  could not read: {e}"]
    out = [f"  shape={df.shape}", f"  columns={list(df.columns)}", "  head:"]
    out += ["    " + line for line in df.head(3).to_string().splitlines()]
    for col in df.columns:
        n = df[col].nunique()
        if n <= 12:
            out.append(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")
        else:
            out.append(f"  {col}: {n} unique values")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    if not root.exists():
        sys.exit(f"{root} does not exist. Run scripts/download_data.sh first.")

    lines = [f"# Inspection of {root}", "", "## Folder layout"] + tree_summary(root)
    lines += ["", "## JSON files"]
    for p in sorted(root.rglob("*.json"))[:40]:
        lines += [f"- {p.relative_to(root)}"] + describe_json(p)
    lines += ["", "## Spreadsheets"]
    for p in sorted([*root.rglob("*.xlsx"), *root.rglob("*.csv")])[:10]:
        lines += [f"- {p.relative_to(root)}"] + describe_table(p)

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nSaved to {args.out}. Paste or commit this so the manifest can be configured.")


if __name__ == "__main__":
    main()
