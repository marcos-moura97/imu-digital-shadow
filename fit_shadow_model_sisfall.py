#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from mpu_shadow.io import load_imu_csv, save_imu_csv
from mpu_shadow.simulator import ShadowConfig, SensorParams
from mpu_shadow.utils import save_json

from mpu_shadow.datasets.sisfall import build_sisfall_dataframe

# ============================================================
# Utility helpers
# ============================================================

def print_sisfall_diagnostics(df: pd.DataFrame):
    print("=== SISFall diagnostics ===")
    print(f"Total rows: {len(df)}")

    if "activity_code" in df.columns:
        print("\nActivity counts:")
        print(df["activity_code"].value_counts())

    gmag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
    wmag = np.sqrt(df["gx_dps"]**2 + df["gy_dps"]**2 + df["gz_dps"]**2)

    print("\nAcceleration magnitude |a| [g]:")
    print(gmag.describe())

    print("\nGyroscope magnitude |ω| [deg/s]:")
    print(wmag.describe())


def quasistatic_subset(
    df: pd.DataFrame,
    gyro_threshold_dps: float = 5.0,
    acc_mag_tol_g: float = 0.10,
    min_run_length: int = 200,
) -> pd.DataFrame:
    """
    Extract quasi-static windows where:
      |ω| < gyro_threshold_dps
      ||a| - 1 g| < acc_mag_tol_g

    Keeps only contiguous runs of at least min_run_length samples.
    """
    gmag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
    wmag = np.sqrt(df["gx_dps"]**2 + df["gy_dps"]**2 + df["gz_dps"]**2)

    base_mask = (np.abs(gmag - 1.0) < acc_mag_tol_g) & (wmag < gyro_threshold_dps)
    base_mask = base_mask.to_numpy(dtype=bool)

    keep = np.zeros_like(base_mask, dtype=bool)
    start = None

    for i, flag in enumerate(base_mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_run_length:
                keep[start:i] = True
            start = None

    if start is not None and len(base_mask) - start >= min_run_length:
        keep[start:] = True

    return df.iloc[np.where(keep)[0]].copy()


def sanity_check_shadow_params(
    acc_noise_density,
    gyr_noise_density,
    acc_rw,
    gyr_rw,
    acc_noise_limit_g_per_sqrtHz: float = 0.1,
    gyr_noise_limit_dps_per_sqrtHz: float = 5.0,
):
    for i, v in enumerate(acc_noise_density):
        if not np.isfinite(v) or v < 0 or v > acc_noise_limit_g_per_sqrtHz:
            raise ValueError(
                f"Accelerometer noise_density[{i}]={v:.4f} is unrealistic "
                "(likely motion leakage from SISFall)."
            )

    for i, v in enumerate(gyr_noise_density):
        if not np.isfinite(v) or v < 0 or v > gyr_noise_limit_dps_per_sqrtHz:
            raise ValueError(
                f"Gyroscope noise_density[{i}]={v:.4f} is unrealistic "
                "(likely motion leakage from SISFall)."
            )

    for i, v in enumerate(acc_rw):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid accelerometer bias_random_walk[{i}]={v}")

    for i, v in enumerate(gyr_rw):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid gyroscope bias_random_walk[{i}]={v}")



def _contiguous_runs(mask: np.ndarray, min_run_length: int):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_run_length]


def fit_from_sisfall(
    sisfall_root,
    output_path,
    sample_rate_hz=200.0,
    max_trials=None,
    save_processed_csv=None,
    gyro_threshold_dps=5.0,
    acc_mag_tol_g=0.10,
    min_run_length=400,
    estimator="diff",
):
    if str(sisfall_root).lower().endswith(".csv"):
        df = load_imu_csv(sisfall_root)
    else:
        df = build_sisfall_dataframe(sisfall_root, max_trials=max_trials,
                                     activity_regex=r"^[DF]\d{2}$")

    if save_processed_csv:
        save_imu_csv(df, save_processed_csv)
        print(f"Saved processed SisFall CSV to {save_processed_csv}")

    dt = 1.0 / sample_rate_hz
    acc_cols = ["ax_g", "ay_g", "az_g"]
    gyr_cols = ["gx_dps", "gy_dps", "gz_dps"]

    group_col = "trial_key" if "trial_key" in df.columns else None
    groups = df.groupby(group_col, sort=False) if group_col else [("all", df)]

    acc_sig, gyr_sig, gyr_bias_runs = [], [], []
    total_samples = 0

    for _, trial in groups:
        a = trial[acc_cols].to_numpy(dtype=float)
        g = trial[gyr_cols].to_numpy(dtype=float)
        gmag = np.linalg.norm(a, axis=1)
        # remove the trial's median rate so the (several deg/s) ITG3200 offset
        # does not eat the stillness threshold
        wmag = np.linalg.norm(g - np.median(g, axis=0), axis=1)
        mask = (np.abs(gmag - 1.0) < acc_mag_tol_g) & (wmag < gyro_threshold_dps)

        for s, e in _contiguous_runs(mask, min_run_length):
            sa, sg = a[s:e], g[s:e]
            total_samples += e - s
            gyr_bias_runs.append(sg.mean(axis=0))
            if estimator == "diff":
                acc_sig.append(np.diff(sa, axis=0).std(axis=0, ddof=1) / np.sqrt(2))
                gyr_sig.append(np.diff(sg, axis=0).std(axis=0, ddof=1) / np.sqrt(2))
            else:
                acc_sig.append(sa.std(axis=0, ddof=1))
                gyr_sig.append(sg.std(axis=0, ddof=1))

    if len(acc_sig) == 0:
        raise ValueError("No quasi-static runs found in SisFall; relax the thresholds")

    print(f"Quasi-static runs used: {len(acc_sig)}  ({total_samples} samples, "
          f"{total_samples * dt / 60:.1f} min)")

    acc_noise_density = (np.median(np.vstack(acc_sig), axis=0) * np.sqrt(dt)).tolist()
    gyr_noise_density = (np.median(np.vstack(gyr_sig), axis=0) * np.sqrt(dt)).tolist()
    gyr_bias = np.median(np.vstack(gyr_bias_runs), axis=0).tolist()

    print(f"Estimator: {estimator}")
    print("  acc noise density [g/sqrt(Hz)]:     ", np.round(acc_noise_density, 6).tolist())
    print("  gyro noise density [dps/sqrt(Hz)]:  ", np.round(gyr_noise_density, 6).tolist())
    print("  gyro bias [dps]:                    ", np.round(gyr_bias, 4).tolist())

    sanity_check_shadow_params(acc_noise_density, gyr_noise_density, [0.0] * 3, [0.0] * 3)

    cfg = ShadowConfig(
        sample_rate_hz=sample_rate_hz,
        accelerometer=SensorParams(
            A=np.eye(3).tolist(),
            bias=[0.0, 0.0, 0.0],
            noise_density=acc_noise_density,
            bias_random_walk=[0.0, 0.0, 0.0],  
            temp_coeff=[0.0, 0.0, 0.0],        
            temp_ref_c=25.0,
        ),
        gyroscope=SensorParams(
            A=np.eye(3).tolist(),
            bias=gyr_bias,
            noise_density=gyr_noise_density,
            bias_random_walk=[0.0, 0.0, 0.0],
            temp_coeff=[0.0, 0.0, 0.0],
            temp_ref_c=25.0,
        ),
    )

    save_json(cfg.to_dict(), output_path)
    print(f"Saved SisFall benchmark shadow config to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Fit MPU6050 digital shadow model (SISFall benchmark mode)"
    )

    ap.add_argument("--sisfall_root", required=True,
                    help="Unified CSV from convert_sisfall.py, or raw SisFall folder")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample_rate_hz", type=float, default=200.0)
    ap.add_argument("--estimator", choices=["diff", "std"], default="diff")
    ap.add_argument("--min_run_length", type=int, default=400)

    args = ap.parse_args()

    fit_from_sisfall(
        sisfall_root=args.sisfall_root,
        output_path=args.output,
        sample_rate_hz=args.sample_rate_hz,
        max_trials=args.max_trials,
        min_run_length=args.min_run_length,
        estimator=args.estimator,
    )


if __name__ == "__main__":
    main()
