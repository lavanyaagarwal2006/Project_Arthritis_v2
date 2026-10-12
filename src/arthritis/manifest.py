"""Build a joint-level manifest (one row per joint ROI) from RAM-H1200.

Two layouts are supported:
  A. Pre-cropped (the RAM-H1200 benchmark layout):
       <split>/_annotation_be_scores.json  +  <split>/<case>/<JOINT>.bmp
  B. Whole images + COCO joint boxes: crops are taken on the fly using a bbox stored in
     the manifest.

The manifest is joined to Metadata.xlsx for patient ID, center and RA status. Patient ID
is mandatory: without it, patient-grouped splits are impossible, so we fail loudly.

Usage: python -m arthritis.manifest [--config configs/base.yaml]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, resolve

BE_JOINTS = {  # crop-file stem -> score key (matches the RAM-H1200 benchmark code)
    "R": "BE_R", "U": "BE_U", "IP": "BE_IP", "L": "BE_L",
    "MCP-T": "BE_MCP-T", "MCP-I": "BE_MCP-I", "MCP-M": "BE_MCP-M", "MCP-R": "BE_MCP-R", "MCP-S": "BE_MCP-S",
    "CMC-T": "BE_CMC-T", "PIP-I": "BE_PIP-I", "PIP-M": "BE_PIP-M", "PIP-R": "BE_PIP-R", "PIP-S": "BE_PIP-S",
    "S": "BE_S", "Tm": "BE_Tm",
}
JSN_JOINTS = {
    "MCP-T": "JSN_MCP-T", "MCP-I": "JSN_MCP-I", "MCP-M": "JSN_MCP-M", "MCP-R": "JSN_MCP-R", "MCP-S": "JSN_MCP-S",
    "PIP-I": "JSN_PIP-I", "PIP-M": "JSN_PIP-M", "PIP-R": "JSN_PIP-R", "PIP-S": "JSN_PIP-S",
    "CMC-M": "JSN_CMC-M", "CMC-R": "JSN_CMC-R", "CMC-S": "JSN_CMC-S", "SC": "JSN_SC", "SR": "JSN_SR", "STT": "JSN_STT",
}
SCORE_FILES = {"BE": "_annotation_be_scores.json", "JSN": "_annotation_jsn_scores.json"}
IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _find_img(folder: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _layout_a(root: Path, score_type: str) -> list[dict]:
    joints = BE_JOINTS if score_type == "BE" else JSN_JOINTS
    rows = []
    for score_path in sorted(root.rglob(SCORE_FILES[score_type])):
        split_dir = score_path.parent
        scores = json.loads(score_path.read_text())
        for hand_name, rec in scores.items():
            case = Path(hand_name).stem
            case_dir = split_dir / case
            for stem, key in joints.items():
                if key not in rec:
                    continue
                img = _find_img(case_dir, stem) if case_dir.exists() else None
                rows.append({
                    "split": split_dir.name, "case": case, "joint": stem, "score": int(rec[key]),
                    "img_path": str(img.relative_to(root)) if img else "",
                    "bbox": "",
                })
    return rows


def _layout_b_fill_bboxes(root: Path, rows: list[dict]) -> None:
    """For rows with no crop file, look for COCO joint-box JSONs and the whole image."""
    missing = [r for r in rows if not r["img_path"]]
    if not missing:
        return
    joint_names = set(BE_JOINTS) | set(JSN_JOINTS)
    index: dict[tuple[str, str], tuple[str, list[float]]] = {}
    for p in root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not (isinstance(obj, dict) and {"images", "annotations", "categories"} <= set(obj)):
            continue
        cats = {c["id"]: c["name"] for c in obj["categories"]}
        if len(joint_names & set(cats.values())) < 8:
            continue
        imgs = {}
        for im in obj["images"]:
            name = (im.get("extra") or {}).get("name") or im["file_name"]
            hits = list(root.rglob(Path(im["file_name"]).name))
            if hits:
                imgs[im["id"]] = (Path(name).stem, str(hits[0].relative_to(root)))
        for a in obj["annotations"]:
            if a["image_id"] in imgs:
                case, rel = imgs[a["image_id"]]
                index.setdefault((case, cats[a["category_id"]]), (rel, a["bbox"]))
    for r in missing:
        hit = index.get((r["case"], r["joint"]))
        if hit:
            r["img_path"], r["bbox"] = hit[0], ",".join(f"{v:.1f}" for v in hit[1])


def _pick_col(df: pd.DataFrame, explicit: str, keywords: list[str]) -> str | None:
    if explicit and explicit != "auto":
        return explicit
    norm = {c: str(c).lower().replace(" ", "").replace("_", "") for c in df.columns}
    for kw in keywords:
        for c, n in norm.items():
            if kw in n:
                return c
    return None


def _join_metadata(root: Path, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    if d["metadata_file"] != "auto":
        meta_path = resolve(d["metadata_file"])
    else:
        cands = sorted(root.rglob("*etadata*.xlsx")) + sorted(root.rglob("*etadata*.csv"))
        if not cands:
            raise SystemExit("No Metadata.xlsx found; set data.metadata_file in configs/base.yaml")
        meta_path = cands[0]
    meta = pd.read_excel(meta_path) if meta_path.suffix == ".xlsx" else pd.read_csv(meta_path)

    pcol = _pick_col(meta, d["metadata_patient_col"], ["patientid", "patient"])
    ccol = _pick_col(meta, d["metadata_center_col"], ["center", "centre", "site", "hospital"])
    rcol = _pick_col(meta, d["metadata_isra_col"], ["isra"])
    icol = d["metadata_image_col"]
    if icol == "auto":  # the column whose values best match case names
        cases = set(df["case"])
        best, best_hits = None, 0
        for c in meta.columns:
            vals = meta[c].astype(str).map(lambda s: Path(s).stem)
            hits = len(cases & set(vals))
            if hits > best_hits:
                best, best_hits = c, hits
        icol = best
    if pcol is None or icol is None:
        raise SystemExit(
            f"Could not identify patient/image columns in {meta_path.name}: columns={list(meta.columns)}. "
            "Set data.metadata_patient_col / metadata_image_col explicitly.")

    m = pd.DataFrame({"case": meta[icol].astype(str).map(lambda s: Path(s).stem),
                      "patient_id": meta[pcol].astype(str)})
    m["center"] = meta[ccol].astype(str) if ccol else "unknown"
    m["is_ra"] = meta[rcol] if rcol else pd.NA
    m = m.drop_duplicates("case")
    out = df.merge(m, on="case", how="left")
    n_unmatched = out["patient_id"].isna().sum()
    if n_unmatched:
        bad = out.loc[out["patient_id"].isna(), "case"].unique()[:5]
        raise SystemExit(f"{n_unmatched} joint rows have no metadata match (e.g. {list(bad)}). "
                         f"Check data.metadata_image_col (auto-picked '{icol}').")
    print(f"metadata: {meta_path.name} image_col='{icol}' patient_col='{pcol}' center_col='{ccol}' isra_col='{rcol}'")
    return out


def build_manifest(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["data"]["root"])
    st = cfg["data"]["score_type"].upper()
    rows = _layout_a(root, st)
    if not rows:
        raise SystemExit(f"No {SCORE_FILES[st]} found under {root}. Run scripts/inspect_dataset.py and adapt.")
    _layout_b_fill_bboxes(root, rows)
    df = pd.DataFrame(rows)
    n_missing = (df["img_path"] == "").sum()
    if n_missing:
        print(f"WARNING: {n_missing}/{len(df)} joints have no image or bbox and are dropped "
              "(logged in splits/exclusions.csv)")
        df.loc[df["img_path"] == ""].assign(reason="no image/bbox").to_csv(
            resolve("splits/exclusions.csv"), index=False)
        df = df[df["img_path"] != ""].copy()
    df["label"] = (df["score"] >= int(cfg["data"]["positive_threshold"])).astype(int)
    df = _join_metadata(root, df, cfg)
    return df.sort_values(["split", "case", "joint"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> str:
    g = df.groupby("split")
    lines = ["split | patients | hands | joints | joint prevalence | hands with any erosion"]
    for s, d in g:
        hand_pos = d.groupby("case")["label"].max()
        lines.append(f"{s} | {d.patient_id.nunique()} | {d.case.nunique()} | {len(d)} | "
                     f"{d.label.mean():.3f} | {int(hand_pos.sum())}/{len(hand_pos)}")
    lines.append(f"score distribution: {df.score.value_counts().sort_index().to_dict()}")
    lines.append(f"centers: {df.groupby('center').patient_id.nunique().to_dict()}")
    leak = df.groupby("patient_id").split.nunique()
    lines.append(f"patients appearing in >1 official split: {int((leak > 1).sum())}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()
    cfg = load_config(args.exp)
    df = build_manifest(cfg)
    out = resolve(cfg["data"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(summarize(df))
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
