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

from mpu_shadow.datasets.sisfall import build_sisfall_dataframe

# ============================================================
# Utility helpers
# ============================================================

def _axis_or_zero(allan, cols, field):
    vals = []
    for c in cols:
        v = getattr(allan[c], field)
        vals.append(0.0 if not np.isfinite(v) else float(v))
    return vals


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



def fit_from_sisfall(
    sisfall_root,
    output_path,
    sample_rate_hz=200.0,
    max_trials=None,
    save_processed_csv=None,
):
    df = load_imu_csv(sisfall_root)

    if save_processed_csv:
        save_imu_csv(df, save_processed_csv)
        print(f"Saved processed SisFall CSV to {save_processed_csv}")

    # Quasi-static selection
    gmag = np.sqrt(df.ax_g**2 + df.ay_g**2 + df.az_g**2)
    wmag = np.sqrt(df.gx_dps**2 + df.gy_dps**2 + df.gz_dps**2)

    mask = (np.abs(gmag - 1.0) < 0.1) & (wmag < 5.0)
    df_qs = df[mask]

    if len(df_qs) < 1000:
        raise ValueError("Not enough quasi-static samples in SisFall")

    acc = df_qs[["ax_g", "ay_g", "az_g"]].to_numpy()
    gyr = df_qs[["gx_dps", "gy_dps", "gz_dps"]].to_numpy()

    acc_mean = acc.mean(axis=0)
    gyr_mean = gyr.mean(axis=0)

    acc_detr = acc - acc_mean
    gyr_detr = gyr - gyr_mean

    dt = 1.0 / sample_rate_hz

    acc_noise_density = (acc_detr.std(axis=0, ddof=1) * np.sqrt(dt)).tolist()
    gyr_noise_density = (gyr_detr.std(axis=0, ddof=1) * np.sqrt(dt)).tolist()

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
            bias=gyr_mean.tolist(),
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

    ap.add_argument("--sisfall_root")
    ap.add_argument("--max_trials", type=int, default=None)

    ap.add_argument("--output", required=True)
    ap.add_argument("--sample_rate_hz", type=float, default=None)

    args = ap.parse_args()

    if args.sisfall_root:
        
        fit_from_sisfall(
                sisfall_root=args.sisfall_root,
                output_path=args.output,
                max_trials=args.max_trials,
            )


if __name__ == "__main__":
    main()
