from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class AllanAxisResult:
    tau: list
    adev: list
    white_noise_density: float
    bias_instability: float
    random_walk: float


def overlapping_allan_deviation(rate_signal: np.ndarray, fs: float, num_points: int = 80):
    rate_signal = np.asarray(rate_signal, dtype=float)
    dt = 1.0 / fs
    theta = np.cumsum(rate_signal) * dt
    L = len(theta)
    max_m = int(2 ** np.floor(np.log2(L / 2)))
    if max_m < 2:
        raise ValueError("Signal too short for Allan deviation")

    m = np.unique(np.ceil(np.logspace(np.log10(1), np.log10(max_m), num_points)).astype(int))
    tau = m * dt
    avar = np.zeros_like(tau, dtype=float)
    for i, mi in enumerate(m):
        avar[i] = np.sum((theta[2*mi:] - 2*theta[mi:-mi] + theta[:-2*mi])**2) / (2 * (tau[i]**2) * (L - 2*mi))
    adev = np.sqrt(avar)
    return tau, adev


def _fit_log_slope_region(tau, adev, target_slope, tol=0.15):
    x = np.log10(tau)
    y = np.log10(adev)
    slopes = np.gradient(y, x)
    idx = np.where(np.abs(slopes - target_slope) < tol)[0]
    if len(idx) == 0:
        return None
    k = idx[np.argmin(np.abs(slopes[idx] - target_slope))]
    return tau[k], adev[k]


def estimate_noise_terms_from_allan(tau, adev):
    wn_pt = _fit_log_slope_region(tau, adev, -0.5)
    N = float(wn_pt[1] * np.sqrt(wn_pt[0])) if wn_pt is not None else float('nan')

    bi_pt = _fit_log_slope_region(tau, adev, 0.0)
    B = float(bi_pt[1] / 0.664) if bi_pt is not None else float('nan')

    rrw_pt = _fit_log_slope_region(tau, adev, +0.5)
    K = float(rrw_pt[1] * np.sqrt(3.0 / rrw_pt[0])) if rrw_pt is not None else float('nan')
    return N, B, K


def analyze_stationary_csv(df: pd.DataFrame, fs: float):
    out = {}
    for col in ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]:
        tau, adev = overlapping_allan_deviation(df[col].to_numpy(), fs)
        N, B, K = estimate_noise_terms_from_allan(tau, adev)
        out[col] = AllanAxisResult(
            tau=tau.tolist(),
            adev=adev.tolist(),
            white_noise_density=float(N),
            bias_instability=float(B),
            random_walk=float(K),
        )
    return out
