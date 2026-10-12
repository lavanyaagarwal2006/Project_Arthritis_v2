"""Cross-validated training that writes out-of-fold (OOF) predictions.

For each outer fold k:
  - outer test  = dev patients in fold k (predicted once, never used for any decision)
  - inner val   = 15% of the remaining patients (early stopping on inner-val AUROC)
  - train       = the rest
Using the outer fold for early stopping would leak selection into the OOF estimate, so
this script never does that.

Schemes:
  --scheme cv    5-fold patient-grouped CV (internal validation)
  --scheme loco  leave-one-center-out (internal-external validation)

Usage:
  python -m arthritis.train --exp densenet121 [--scheme cv|loco] [--smoke]
  python -m arthritis.train --exp majority                # baseline, no torch needed
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import git_sha, load_config, resolve
from .splits import attach


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def outer_folds(dev: pd.DataFrame, scheme: str) -> list[tuple[str, np.ndarray]]:
    if scheme == "cv":
        return [(str(k), (dev.fold == k).values) for k in sorted(dev.fold.unique())]
    if scheme == "loco":
        return [(str(c), (dev.center == c).values) for c in sorted(dev.center.unique())]
    raise ValueError(scheme)


def inner_split(train: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    pts = train.patient_id.unique()
    rng = np.random.default_rng(seed)
    val_pts = set(rng.choice(pts, max(1, int(round(frac * len(pts)))), replace=False))
    return train.patient_id.isin(val_pts).values


# ---------------------------------------------------------------- baseline (no torch)
def run_majority(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train.label.mean())


def run_pixel_logreg(train: pd.DataFrame, test: pd.DataFrame, root: Path, size: int = 32) -> np.ndarray:
    """Classical rung: L2 logistic regression on 32x32 normalised pixels. Shows how much deep learning adds."""
    from PIL import Image
    from sklearn.linear_model import LogisticRegressionCV

    def feats(df):
        from .imaging import letterbox, load_roi
        return np.stack([np.asarray(Image.fromarray((letterbox(load_roi(root, r.img_path, getattr(r, "bbox", None))) * 255)
                                                    .astype(np.uint8)).resize((size, size)), np.float32).ravel() / 255
                         for r in df.itertuples()])
    clf = LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, scoring="roc_auc", class_weight="balanced")
    clf.fit(feats(train), train.label.values)
    return clf.predict_proba(feats(test))[:, 1]


# ---------------------------------------------------------------- torch training
def run_torch(cfg: dict, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, root: Path,
              ckpt_path: Path, log) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from .data import JointDataset, balanced_sampler, build_transforms
    from .metrics import auroc
    from .models import build_model

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tcfg = cfg["train"]
    model, ch, pretrained = build_model(cfg["model"])
    model.to(dev)

    tr_ds = JointDataset(train, root, build_transforms(cfg, True, pretrained), ch)
    ev_tf = build_transforms(cfg, False, pretrained)
    nw = tcfg["num_workers"]
    sampler = balanced_sampler(train.label.values) if tcfg["balanced_sampling"] else None
    tr_dl = DataLoader(tr_ds, batch_size=tcfg["batch_size"], sampler=sampler, shuffle=sampler is None,
                       num_workers=nw, drop_last=True, pin_memory=dev == "cuda")

    def predict(df):
        dl = DataLoader(JointDataset(df, root, ev_tf, ch), batch_size=tcfg["batch_size"] * 2, num_workers=nw)
        model.eval(); out = []
        with torch.no_grad():
            for x, _, _ in dl:
                out.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
        return np.concatenate(out)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = max(1, len(tr_dl)) * tcfg["epochs"]
    warm = max(1, len(tr_dl)) * tcfg["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    # balanced sampling already equalises classes; otherwise weight positives
    pos_w = None if sampler else torch.tensor((1 - train.label.mean()) / max(train.label.mean(), 1e-6), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    use_amp = tcfg["amp"] and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad = -1.0, 0
    for ep in range(tcfg["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y, _ in tr_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y)
        v_auc = auroc(val.label.values, predict(val))
        if not np.isfinite(v_auc):  # inner-val has one class (tiny smoke subsets): keep training, save last epoch
            v_auc = float(ep) * 1e-3
        log(f"    epoch {ep:02d} loss {tot / max(1, len(tr_dl.dataset)):.4f} inner-val AUROC {v_auc:.4f} ({time.time() - t0:.0f}s)")
        if v_auc > best + 1e-4:
            best, bad = v_auc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= tcfg["early_stop_patience"]:
                log(f"    early stop at epoch {ep} (best inner-val AUROC {best:.4f})")
                break
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    return predict(test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--scheme", default="cv", choices=["cv", "loco"])
    ap.add_argument("--smoke", action="store_true", help="2 epochs, small subset: checks the pipeline, not the model")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.exp)
    if args.smoke:
        cfg["train"].update(epochs=2, warmup_epochs=1, early_stop_patience=2, num_workers=0, batch_size=16)
    seed_everything(cfg["seed"])
    root = resolve(cfg["data"]["root"])
    man = pd.read_csv(resolve(cfg["data"]["manifest"]), dtype={"patient_id": str})
    folds = pd.read_csv(resolve(cfg["data"]["folds"]), dtype={"patient_id": str})
    man = attach(man, folds)
    dev_df = man[man.role == "dev"].reset_index(drop=True)
    assert (dev_df.role == "dev").all() and not set(dev_df.patient_id) & set(man.loc[man.role == "sealed", "patient_id"]), \
        "sealed-test patients leaked into dev"
    if args.smoke:
        keep = dev_df.patient_id.drop_duplicates().sample(frac=0.3, random_state=0)
        dev_df = dev_df[dev_df.patient_id.isin(keep)].reset_index(drop=True)

    run_name = args.run_name or f"{datetime.now():%Y%m%d-%H%M}_{cfg['exp_name']}_{args.scheme}{'_smoke' if args.smoke else ''}"
    run_dir = resolve("results") / run_name
    (run_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = resolve("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.log", "a")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    meta = {"exp_name": cfg["exp_name"], "scheme": args.scheme, "git_sha": git_sha(), "smoke": args.smoke,
            "started": datetime.now().isoformat(timespec="seconds"), "config": cfg}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"run {run_name}: {len(dev_df)} dev joints / {dev_df.patient_id.nunique()} patients, git {meta['git_sha']}")

    preds = np.full(len(dev_df), np.nan)
    outer_id = np.empty(len(dev_df), dtype=object)
    for name, te_mask in outer_folds(dev_df, args.scheme):
        tr_all, te = dev_df[~te_mask], dev_df[te_mask]
        if te.label.nunique() < 2:
            log(f"  fold {name}: test fold has one class only; predictions kept, AUROC undefined for this fold")
        assert not set(tr_all.patient_id) & set(te.patient_id), "patient overlap between train and test fold"
        log(f"  outer fold {name}: train+val {tr_all.patient_id.nunique()} pts, test {te.patient_id.nunique()} pts")
        if cfg["model"]["name"] == "majority":
            p = run_majority(tr_all, te)
        elif cfg["model"]["name"] == "pixel_logreg":
            p = run_pixel_logreg(tr_all, te, root)
        else:
            vmask = inner_split(tr_all, 0.15, cfg["seed"])
            p = run_torch(cfg, tr_all[~vmask], tr_all[vmask], te, root, ckpt_dir / f"fold_{name}.pt", log)
        preds[te_mask] = p
        outer_id[te_mask] = name

    assert np.isfinite(preds).all(), "some dev joints were never predicted"
    out = dev_df[["case", "joint", "patient_id", "center", "score", "label"]].copy()
    out["fold"] = outer_id
    out["prob"] = preds
    out.to_csv(run_dir / "oof_predictions.csv", index=False)
    log(f"wrote {run_dir / 'oof_predictions.csv'}; next: python -m arthritis.evaluate {run_dir.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
