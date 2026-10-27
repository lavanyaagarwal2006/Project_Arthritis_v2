"""Torch dataset for joint ROI crops.

Preprocessing is identical at train and test time (only augmentation differs):
  1. Load grayscale; crop by bbox if the manifest gives one.
  2. Robust intensity normalisation: 1st–99th percentile -> [0, 1].
  3. Optional: mirror left hands so every crop has right-hand orientation (train.mirror_left).
  4. Geometry, by train.input_mode:
       resize : letterbox to a square, resize to image_size (v1; mixes physical scales between joints)
       native : keep native pixels on a canvas x canvas square (v2; 1 px = same mm everywhere)
  5. Replicate to 3 channels, ImageNet normalisation for pretrained backbones.

Each item: (image, binary label, grade index, joint index, task id, row index).
task 0 = erosion (BE, the endpoint); task 1 = joint-space narrowing (JSN, auxiliary).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from .imaging import fit_canvas, letterbox, load_roi
from .manifest import BE_JOINTS, JSN_JOINTS

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
JOINT_VOCAB = [f"BE:{j}" for j in BE_JOINTS] + [f"JSN:{j}" for j in JSN_JOINTS]
JOINT_INDEX = {k: i for i, k in enumerate(JOINT_VOCAB)}
TASKS = {"BE": 0, "JSN": 1}


def build_transforms(cfg: dict, train: bool, pretrained: bool):
    t = cfg["train"]
    aug = t["augment"]
    native = t.get("input_mode", "resize") == "native"
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=False)]
    if native:
        if train:
            ops += [T.RandomAffine(degrees=aug["rotate_deg"], translate=(aug["translate"], aug["translate"]),
                                   scale=(0.92, 1.08)),
                    T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"])]
        # Optional zoom: the canvas keeps one physical scale for every joint; resizing the whole canvas
        # afterwards keeps that scale uniform while giving small (finger) erosions more pixels in the network.
        if int(t.get("image_size", t.get("canvas", 224))) != int(t.get("canvas", 224)):
            ops.append(T.Resize((int(t["image_size"]), int(t["image_size"])), antialias=True))
    else:
        size = t["image_size"]
        if train:
            ops += [T.RandomResizedCrop(size, scale=tuple(aug["scale"]), ratio=(0.9, 1.1), antialias=True),
                    T.RandomRotation(aug["rotate_deg"]),
                    T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"])]
        else:
            ops.append(T.Resize((size, size), antialias=True))
    if train and aug.get("hflip"):
        ops.append(T.RandomHorizontalFlip())
    if pretrained:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


def prepare_rows(df: pd.DataFrame, task: str, levels: list[int]) -> pd.DataFrame:
    """Adds the integer columns the dataset needs: task_id, joint_idx, grade_idx."""
    from .ordinal import level_index
    d = df.copy()
    d["task_id"] = TASKS[task]
    d["joint_idx"] = d["joint"].map(lambda j: JOINT_INDEX[f"{task}:{j}"]).astype(int)
    d["grade_idx"] = level_index(d["score"].values, levels).astype(int)
    return d


class JointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform, channels: int = 3, cfg: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.channels = channels
        t = (cfg or {}).get("train", {})
        self.native = t.get("input_mode", "resize") == "native"
        self.canvas = int(t.get("canvas", 320))
        self.mirror_left = bool(t.get("mirror_left", False))
        for c, default in (("task_id", 0), ("joint_idx", 0), ("grade_idx", None)):
            if c not in self.df:
                self.df[c] = self.df["label"] if default is None else default

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = load_roi(self.root, r.img_path, r.get("bbox"))
        if self.mirror_left and str(r.get("side", "")) == "L":
            a = a[:, ::-1].copy()
        a = fit_canvas(a, self.canvas)[0] if self.native else letterbox(a)
        a = np.repeat(a[:, :, None], self.channels, axis=2)  # HWC
        x = self.transform(a)
        return (x, torch.tensor(float(r.label)), torch.tensor(int(r.grade_idx)),
                torch.tensor(int(r.joint_idx)), torch.tensor(int(r.task_id)), i)


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)


def multitask_sampler(df: pd.DataFrame, balanced: bool, jsn_share: float) -> WeightedRandomSampler:
    """Erosion rows class-balanced (if requested) and summing to 1; JSN rows uniform, summing to jsn_share.
    One epoch = (#erosion rows) * (1 + jsn_share) draws."""
    be = (df.task_id == 0).values
    w = np.zeros(len(df))
    lab = df.label.values.astype(int)
    if balanced:
        cnt = np.bincount(lab[be], minlength=2).astype(float)
        w[be] = 1.0 / cnt[lab[be]]
    else:
        w[be] = 1.0
    w[be] /= w[be].sum()
    if (~be).any():
        w[~be] = jsn_share / (~be).sum()
    n = int(be.sum() * (1 + (jsn_share if (~be).any() else 0)))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n, replacement=True)
