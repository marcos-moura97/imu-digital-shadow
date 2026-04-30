#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from mpu_shadow.io import load_imu_csv
from mpu_shadow.calibration import fit_accel_six_position, estimate_gyro_bias
from mpu_shadow.allan import analyze_stationary_csv
from mpu_shadow.temperature import fit_linear_temp_bias
from mpu_shadow.simulator import ShadowConfig, SensorParams
from mpu_shadow.utils import save_json


def _axis_or_zero(allan, cols, field):
    vals = []
    for c in cols:
        v = getattr(allan[c], field)
        vals.append(0.0 if not np.isfinite(v) else float(v))
    return vals


def main():
    ap = argparse.ArgumentParser(description='Fit MPU6050 digital shadow model from logged data')
    ap.add_argument('--stationary', required=True)
    ap.add_argument('--sixpos', required=True)
    ap.add_argument('--warmup', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sample_rate_hz', type=float, default=100.0)
    args = ap.parse_args()

    df_stationary = load_imu_csv(args.stationary)
    df_six = load_imu_csv(args.sixpos)
    df_warm = load_imu_csv(args.warmup)

    acc_cal = fit_accel_six_position(df_six)
    gyro_bias = estimate_gyro_bias(df_stationary)
    allan = analyze_stationary_csv(df_stationary, fs=args.sample_rate_hz)
    acc_temp = fit_linear_temp_bias(df_warm, ['ax_g', 'ay_g', 'az_g'])
    gyr_temp = fit_linear_temp_bias(df_warm, ['gx_dps', 'gy_dps', 'gz_dps'])

    cfg = ShadowConfig(
        sample_rate_hz=args.sample_rate_hz,
        accelerometer=SensorParams(
            A=acc_cal.A,
            bias=acc_cal.bias,
            noise_density=_axis_or_zero(allan, ['ax_g','ay_g','az_g'], 'white_noise_density'),
            bias_random_walk=_axis_or_zero(allan, ['ax_g','ay_g','az_g'], 'random_walk'),
            temp_coeff=acc_temp.coeffs,
            temp_ref_c=acc_temp.ref_temp_c,
        ),
        gyroscope=SensorParams(
            A=np.eye(3).tolist(),
            bias=gyro_bias.tolist(),
            noise_density=_axis_or_zero(allan, ['gx_dps','gy_dps','gz_dps'], 'white_noise_density'),
            bias_random_walk=_axis_or_zero(allan, ['gx_dps','gy_dps','gz_dps'], 'random_walk'),
            temp_coeff=gyr_temp.coeffs,
            temp_ref_c=gyr_temp.ref_temp_c,
        )
    )

    save_json(cfg.to_dict(), args.output)
    print(f"Saved fitted shadow config to {args.output}")
    print(f"Accelerometer calibration RMS error: {acc_cal.rms_error:.6f}")

if __name__ == '__main__':
    main()
