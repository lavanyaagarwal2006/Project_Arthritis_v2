"""Torch dataset for joint ROI crops.

Preprocessing is identical at train and test time (only the augmentation differs):
  1. Load grayscale; crop by bbox if the manifest gives one (layout B).
  2. Robust intensity normalisation: clip to the 1st–99th percentile, then rescale to [0, 1].
  3. Letterbox to a square (no aspect distortion; erosions are shape features), then resize.
  4. Replicate to 3 channels and apply ImageNet normalisation for pretrained backbones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from .imaging import letterbox, load_roi

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(cfg: dict, train: bool, pretrained: bool):
    size = cfg["train"]["image_size"]
    aug = cfg["train"]["augment"]
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=False)]
    if train:
        ops += [T.RandomResizedCrop(size, scale=tuple(aug["scale"]), ratio=(0.9, 1.1), antialias=True),
                T.RandomRotation(aug["rotate_deg"]),
                T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"])]
        if aug.get("hflip"):
            ops.append(T.RandomHorizontalFlip())
    else:
        ops.append(T.Resize((size, size), antialias=True))
    if pretrained:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


class JointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform, channels: int = 3):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.channels = channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = letterbox(load_roi(self.root, r.img_path, r.get("bbox")))
        a = np.repeat(a[:, :, None], self.channels, axis=2)  # HWC
        x = self.transform(a)
        return x, torch.tensor(float(r.label)), i


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)
