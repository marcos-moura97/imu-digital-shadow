from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

POSE_TO_TRUE = {
    "xp": np.array([+1.0, 0.0, 0.0]),
    "xn": np.array([-1.0, 0.0, 0.0]),
    "yp": np.array([0.0, +1.0, 0.0]),
    "yn": np.array([0.0, -1.0, 0.0]),
    "zp": np.array([0.0, 0.0, +1.0]),
    "zn": np.array([0.0, 0.0, -1.0]),
}

@dataclass
class TriadCalibrationResult:
    A: list
    bias: list
    rms_error: float


def _pack(A, b):
    return np.concatenate([A.ravel(), b])


def _unpack(x):
    A = x[:9].reshape(3, 3)
    b = x[9:12]
    return A, b


def fit_affine_triad(measured: np.ndarray, truth: np.ndarray) -> TriadCalibrationResult:
    measured = np.asarray(measured, dtype=float)
    truth = np.asarray(truth, dtype=float)
    x0 = _pack(np.eye(3), measured.mean(axis=0))

    def residuals(x):
        A, b = _unpack(x)
        corr = (A @ (measured - b).T).T
        return (corr - truth).ravel()

    res = least_squares(residuals, x0, method='trf')
    A, b = _unpack(res.x)
    rms = np.sqrt(np.mean(res.fun**2))
    return TriadCalibrationResult(A=A.tolist(), bias=b.tolist(), rms_error=float(rms))


def fit_accel_six_position(df: pd.DataFrame, pose_col: str = 'pose_label') -> TriadCalibrationResult:
    if pose_col not in df.columns:
        raise ValueError(f"CSV must contain a '{pose_col}' column for six-position calibration.")

    rows, truths = [], []
    for pose, truth in POSE_TO_TRUE.items():
        sub = df[df[pose_col] == pose]
        if len(sub) == 0:
            continue
        mean_meas = sub[["ax_g", "ay_g", "az_g"]].mean().to_numpy(dtype=float)
        rows.append(mean_meas)
        truths.append(truth)

    if len(rows) < 6:
        raise ValueError("Need all six poses (xp, xn, yp, yn, zp, zn).")

    measured = np.vstack(rows)
    truth = np.vstack(truths)
    return fit_affine_triad(measured, truth)


def estimate_gyro_bias(df: pd.DataFrame) -> np.ndarray:
    return df[["gx_dps", "gy_dps", "gz_dps"]].mean().to_numpy(dtype=float)
