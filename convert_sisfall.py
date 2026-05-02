#!/usr/bin/env python3
from __future__ import annotations

import argparse
from mpu_shadow.datasets.sisfall import build_sisfall_dataframe
from mpu_shadow.io import save_imu_csv


def main():
    ap = argparse.ArgumentParser(
        description="Convert SisFall dataset into unified IMU CSV format"
    )
    ap.add_argument("--sisfall_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument(
        "--activity_regex",
        default=None,
        help="Regex filter, e.g. D0[1-9]|F0[1-9]"
    )
    args = ap.parse_args()

    df = build_sisfall_dataframe(
        args.sisfall_root,
        max_trials=args.max_trials,
        activity_regex=args.activity_regex,
    )
    save_imu_csv(df, args.output)
    print(f"Saved SisFall unified CSV: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()