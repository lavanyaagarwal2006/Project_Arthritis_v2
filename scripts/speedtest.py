"""60-second check of where training time goes on this machine. Prints device, memory, model step speed
and data-loading speed, then an estimated time per epoch.

Usage: python scripts/speedtest.py [--exp densenet121] [--batch 32]
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, "src")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="densenet121")
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from arthritis.config import load_config, resolve
    from arthritis.data import JointDataset, build_transforms, prepare_rows
    from arthritis.models import build_model

    cfg = load_config(args.exp)
    bs = args.batch or cfg["train"]["batch_size"]
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    except Exception:
        mem = float("nan")
    print(f"python {platform.python_version()} | torch {torch.__version__} | {platform.machine()} | RAM {mem:.0f} GB")
    print(f"MPS available: {torch.backends.mps.is_available()} | CUDA: {torch.cuda.is_available()}")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"training would use: {dev.upper()}  (exp={cfg['exp_name']}, batch={bs})")

    model, ch, pre, _, _ = build_model(cfg)
    model.to(dev).train()
    side = cfg["train"]["canvas"] if cfg["train"].get("input_mode") == "native" else cfg["train"]["image_size"]
    x = torch.randn(bs, ch, side, side, device=dev)
    j = torch.zeros(bs, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    for i in range(6):
        if i == 2:
            if dev == "mps": torch.mps.synchronize()
            t0 = time.time()
        be, _ = model(x, j)
        loss = be.float().mean(); opt.zero_grad(); loss.backward(); opt.step()
    if dev == "mps": torch.mps.synchronize()
    if dev == "cuda": torch.cuda.synchronize()
    step = (time.time() - t0) / 4
    print(f"model: {step:.2f} s per training step of {bs} images  -> {bs / step:.0f} img/s")

    man = pd.read_csv(resolve(cfg["data"]["manifest"]), dtype={"patient_id": str}).head(bs * 8)
    ds = JointDataset(prepare_rows(man, "BE", cfg["levels"]["BE"]), resolve(cfg["data"]["root"]),
                      build_transforms(cfg, True, pre), ch, cfg)
    for nw in (0, 4):
        dl = DataLoader(ds, batch_size=bs, num_workers=nw)
        t0 = time.time()
        for _ in dl:
            pass
        rate = len(ds) / (time.time() - t0)
        print(f"data loading, {nw} workers: {rate:.0f} img/s (includes worker start-up)")
    n_train = 10_100
    print(f"\nestimated epoch time (~{n_train} crops): ~{n_train / (bs / step) / 60:.0f} min if the model is the bottleneck")
    print("rule of thumb: >15 min/epoch on a Mac -> use the Kaggle GPU route (docs/KAGGLE.md)")


if __name__ == "__main__":
    main()
