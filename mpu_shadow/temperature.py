from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class TemperatureFitResult:
    coeffs: list
    ref_temp_c: float
    axis_names: list


def fit_linear_temp_bias(df: pd.DataFrame, value_cols, temp_col='temp_c') -> TemperatureFitResult:
    t = df[temp_col].to_numpy(dtype=float)
    tref = float(np.mean(t))
    x = (t - tref).reshape(-1, 1)
    coeffs = []
    for c in value_cols:
        y = df[c].to_numpy(dtype=float)
        k = np.linalg.lstsq(x, y, rcond=None)[0][0]
        coeffs.append(float(k))
    return TemperatureFitResult(coeffs=coeffs, ref_temp_c=tref, axis_names=list(value_cols))
