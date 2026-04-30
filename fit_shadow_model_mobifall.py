#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from mpu_shadow.io import load_imu_csv, save_imu_csv
from mpu_shadow.calibration import fit_accel_six_position, estimate_gyro_bias
from mpu_shadow.allan import analyze_stationary_csv
from mpu_shadow.temperature import fit_linear_temp_bias
from mpu_shadow.simulator import ShadowConfig, SensorParams
from mpu_shadow.utils import save_json


from mpu_shadow.datasets.mobifall import (
    build_mobifall_dataframe,
    infer_sample_rate_hz,
    split_by_trial,
    quasistatic_mask,
    contiguous_runs,
)



# ============================================================
# Utility helpers
# ============================================================

def _axis_or_zero(allan, cols, field):
    vals = []
    for c in cols:
        v = getattr(allan[c], field)
        vals.append(0.0 if not np.isfinite(v) else float(v))
    return vals


def print_mobifall_diagnostics(df: pd.DataFrame):
    print("=== MobiFall diagnostics ===")
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
                "(likely motion leakage from MobiFall)."
            )

    for i, v in enumerate(gyr_noise_density):
        if not np.isfinite(v) or v < 0 or v > gyr_noise_limit_dps_per_sqrtHz:
            raise ValueError(
                f"Gyroscope noise_density[{i}]={v:.4f} is unrealistic "
                "(likely motion leakage from MobiFall)."
            )

    for i, v in enumerate(acc_rw):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid accelerometer bias_random_walk[{i}]={v}")

    for i, v in enumerate(gyr_rw):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid gyroscope bias_random_walk[{i}]={v}")

# ============================================================
# MobiFall benchmark fitting
# ============================================================


def fit_from_mobifall(
    mobifall_root,
    output_path,
    sample_rate_hz=None,
    max_trials=None,
    save_processed_csv=None,
    gyro_threshold_dps=5.0,
    acc_mag_tol_g=0.10,
    min_run_length=400,
    max_noise_acc=0.1,
    max_noise_gyr=5.0,
):
    """
    Conservative benchmark fitting from MobiFall.

    Fits only:
      - sample rate
      - gyro bias
      - white-noise densities

    Does NOT fit:
      - accelerometer deterministic calibration
      - temperature coefficients
      - random walk / bias instability
    """
    df = build_mobifall_dataframe(mobifall_root, max_trials=max_trials)

    if save_processed_csv:
        save_imu_csv(df, save_processed_csv)
        print(f"Saved processed MobiFall CSV to {save_processed_csv}")

    if sample_rate_hz is None:
        sample_rate_hz = infer_sample_rate_hz(df)

    dt = 1.0 / sample_rate_hz
    print(f"Using sample rate: {sample_rate_hz:.6f} Hz")

    acc_noise_runs = []
    gyr_noise_runs = []
    gyro_bias_runs = []

    total_runs = 0
    total_samples = 0

    for trial_df in split_by_trial(df):
        mask = quasistatic_mask(
            trial_df,
            gyro_threshold_dps=gyro_threshold_dps,
            acc_mag_tol_g=acc_mag_tol_g,
        )
        runs = contiguous_runs(mask, min_run_length=min_run_length)

        for start, end in runs:
            seg = trial_df.iloc[start:end].copy()
            total_runs += 1
            total_samples += len(seg)

            # per-run means
            acc_mean = seg[["ax_g", "ay_g", "az_g"]].mean().to_numpy(dtype=float)
            gyr_mean = seg[["gx_dps", "gy_dps", "gz_dps"]].mean().to_numpy(dtype=float)

            # gyro bias estimate from quasi-static mean
            gyro_bias_runs.append(gyr_mean)

            # detrend / remove gravity projection and static bias
            acc_detr = seg[["ax_g", "ay_g", "az_g"]].to_numpy(dtype=float) - acc_mean
            gyr_detr = seg[["gx_dps", "gy_dps", "gz_dps"]].to_numpy(dtype=float) - gyr_mean

            # estimate white noise density from sample std:
            # sigma_density ≈ sigma_sample * sqrt(dt)
            acc_std = acc_detr.std(axis=0, ddof=1)
            gyr_std = gyr_detr.std(axis=0, ddof=1)

            acc_noise_runs.append(acc_std * np.sqrt(dt))
            gyr_noise_runs.append(gyr_std * np.sqrt(dt))

    if total_runs == 0:
        raise ValueError(
            "No quasi-static runs found in MobiFall. "
            "Try increasing gyro_threshold_dps or acc_mag_tol_g."
        )

    print(f"Quasi-static runs used: {total_runs}")
    print(f"Total quasi-static samples used: {total_samples}")

    acc_noise_density = np.median(np.vstack(acc_noise_runs), axis=0)
    gyr_noise_density = np.median(np.vstack(gyr_noise_runs), axis=0)
    gyro_bias = np.median(np.vstack(gyro_bias_runs), axis=0)

    print("Estimated benchmark white-noise densities:")
    print("  accelerometer:", acc_noise_density.tolist())
    print("  gyroscope:", gyr_noise_density.tolist())
    print("Estimated benchmark gyro bias:")
    print("  gyro_bias:", gyro_bias.tolist())

    # sanity checks
    if np.any(acc_noise_density > max_noise_acc):
        raise ValueError(
            f"Accelerometer white-noise density too large: {acc_noise_density.tolist()} "
            f"(threshold {max_noise_acc})"
        )

    if np.any(gyr_noise_density > max_noise_gyr):
        raise ValueError(
            f"Gyroscope white-noise density too large: {gyr_noise_density.tolist()} "
            f"(threshold {max_noise_gyr})"
        )

    cfg = ShadowConfig(
        sample_rate_hz=sample_rate_hz,
        accelerometer=SensorParams(
            A=np.eye(3).tolist(),
            bias=[0.0, 0.0, 0.0],
            noise_density=acc_noise_density.tolist(),
            bias_random_walk=[0.0, 0.0, 0.0],   # disabled in benchmark mode
            temp_coeff=[0.0, 0.0, 0.0],
            temp_ref_c=25.0,
        ),
        gyroscope=SensorParams(
            A=np.eye(3).tolist(),
            bias=gyro_bias.tolist(),
            noise_density=gyr_noise_density.tolist(),
            bias_random_walk=[0.0, 0.0, 0.0],   # disabled in benchmark mode
            temp_coeff=[0.0, 0.0, 0.0],
            temp_ref_c=25.0,
        ),
    )

    save_json(cfg.to_dict(), output_path)
    print(f"✅ Saved MobiFall benchmark shadow config to {output_path}")


# ============================================================
# CLI entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fit MPU6050 digital shadow model (MobiFall mode)"
    )

    ap.add_argument("--mobifall_root")
    ap.add_argument("--save_processed_mobifall_csv")
    ap.add_argument("--max_trials", type=int, default=None)

    ap.add_argument("--output", required=True)
    ap.add_argument("--sample_rate_hz", type=float, default=None)

    args = ap.parse_args()

    fit_from_mobifall(
        mobifall_root=args.mobifall_root,
        output_path=args.output,
        sample_rate_hz=args.sample_rate_hz,
        max_trials=args.max_trials,
        save_processed_csv=args.save_processed_mobifall_csv,
    )
    


if __name__ == "__main__":
    main()
