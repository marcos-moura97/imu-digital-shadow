#!/usr/bin/env python3
from __future__ import annotations

import argparse
from mpu_shadow.datasets.mobifall import build_mobifall_dataframe
from mpu_shadow.io import save_imu_csv


def main():
    ap = argparse.ArgumentParser(
        description="Convert raw MobiFall folder into the unified CSV schema used by the shadow pipeline"
    )
    ap.add_argument("--mobifall_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument(
        "--activity_regex",
        default=None,
        help='Optional regex to filter activities, e.g. "STD|WAL|FOL"',
    )
    args = ap.parse_args()

    df = build_mobifall_dataframe(
        args.mobifall_root,
        max_trials=args.max_trials,
        activity_regex=args.activity_regex,
    )
    save_imu_csv(df, args.output)
    print(f"Saved unified MobiFall CSV to {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()