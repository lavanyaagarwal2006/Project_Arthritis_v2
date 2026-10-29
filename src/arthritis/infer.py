"""Inference for trained members: load a checkpoint, predict joint crops. Shared by the sealed-test
evaluation and the web app, so both run exactly the same preprocessing as training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_member(exp: str, ckpt: str | Path, device: str | None = None):
    import torch

    from .config import load_config
    from .models import build_model
    cfg = load_config(exp)
    dev = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, ch, pretrained, k_be, _ = build_model(cfg)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state)
    model.to(dev).eval()
    return {"exp": exp, "cfg": cfg, "model": model, "channels": ch, "pretrained": pretrained, "k_be": k_be, "device": dev}


def predict_member(member: dict, df: pd.DataFrame, root: Path, batch_size: int = 128) -> np.ndarray:
    """Returns [N, K] threshold probabilities P(grade > level_k); column 0 = P(erosion)."""
    import torch
    from torch.utils.data import DataLoader

    from .data import JointDataset, build_transforms, prepare_rows
    cfg = member["cfg"]
    rows = prepare_rows(df, "BE", cfg["levels"]["BE"])
    ds = JointDataset(rows, root, build_transforms(cfg, False, member["pretrained"]), member["channels"], cfg)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=cfg["train"].get("num_workers", 0))
    out = []
    with torch.no_grad():
        for x, _, _, j, _, _ in dl:
            be, _ = member["model"](x.to(member["device"]), j.to(member["device"]))
            out.append(torch.sigmoid(be.float()).cpu().numpy())
    return np.concatenate(out)
