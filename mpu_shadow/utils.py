from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np


def ensure_array(x, shape=None, dtype=float):
    arr = np.array(x, dtype=dtype)
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
