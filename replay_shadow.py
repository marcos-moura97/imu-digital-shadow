#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from mpu_shadow.io import load_imu_csv, save_imu_csv
from mpu_shadow.utils import load_json
from mpu_shadow.simulator import ShadowConfig, ShadowIMU


def main():
    ap = argparse.ArgumentParser(description='Replay a truth log through the digital shadow')
    ap.add_argument('--input', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    df = load_imu_csv(args.input)
    cfg = ShadowConfig.from_dict(load_json(args.config))
    imu = ShadowIMU(cfg, rng=np.random.default_rng(args.seed))
    out = imu.simulate_dataframe(df)
    save_imu_csv(out, args.output)
    print(f"Shadowed replay saved to {args.output}")

if __name__ == '__main__':
    main()
