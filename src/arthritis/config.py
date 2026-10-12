"""Config loading: configs/base.yaml deep-merged with an experiment file."""
from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(exp: str | None = None, base: str = "configs/base.yaml") -> dict:
    cfg = yaml.safe_load((REPO_ROOT / base).read_text())
    if exp:
        p = Path(exp)
        if not p.suffix:
            p = REPO_ROOT / "configs" / "exp" / f"{exp}.yaml"
        elif not p.is_absolute():
            p = REPO_ROOT / p
        cfg = _deep_merge(cfg, yaml.safe_load(p.read_text()))
        cfg["exp_name"] = p.stem
    return cfg


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
