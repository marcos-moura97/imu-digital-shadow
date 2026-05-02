from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# From SISFALL paper: https://www.mdpi.com/1424-8220/17/1/198
SISFALL_FS_HZ = 200.0

# Sensor specs
ADXL345_RANGE_G = 16.0
ADXL345_BITS = 13

ITG3200_RANGE_DPS = 2000.0
ITG3200_BITS = 16


def _subject_from_path(path: Path) -> str:
    for p in path.parts:
        if re.match(r"[Ss][AE]\d{2}", p):
            return p
    return "unknown"


def _activity_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]


def _trial_key(path: Path) -> str:
    return path.stem


def discover_sisfall_files(root: str | Path) -> List[Path]:
    files = sorted(Path(root).rglob("*.txt"))
    if not files:
        raise ValueError("No SisFall .txt files found")
    return files


def _parse_line(line: str) -> Optional[List[int]]:
    nums = re.findall(r"-?\d+", line)
    if len(nums) < 9:
        return None
    return [int(n) for n in nums[:9]]


def read_sisfall_txt(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise ValueError(f"No valid data in {path}")

    raw = np.asarray(rows, dtype=float)

    # Channels
    adxl = raw[:, 0:3]
    gyro = raw[:, 3:6]

    # Convert to physical units
    acc_scale = (2 * ADXL345_RANGE_G) / (2 ** ADXL345_BITS)
    gyr_scale = (2 * ITG3200_RANGE_DPS) / (2 ** ITG3200_BITS)

    acc_g = adxl * acc_scale
    gyr_dps = gyro * gyr_scale

    t = np.arange(len(acc_g)) / SISFALL_FS_HZ

    df = pd.DataFrame({
        "timestamp_s": t,
        "ax_g": acc_g[:, 0],
        "ay_g": acc_g[:, 1],
        "az_g": acc_g[:, 2],
        "gx_dps": gyr_dps[:, 0],
        "gy_dps": gyr_dps[:, 1],
        "gz_dps": gyr_dps[:, 2],
        "temp_c": np.full(len(t), 25.0),
    })

    df["subject_id"] = _subject_from_path(path)
    df["activity_code"] = _activity_from_filename(path)
    df["trial_key"] = _trial_key(path)

    return df

def build_sisfall_dataframe(
    root: str | Path,
    max_trials: Optional[int] = None,
    activity_regex: Optional[str] = None,
) -> pd.DataFrame:
    files = discover_sisfall_files(root)
    if max_trials is not None:
        files = files[:max_trials]

    rgx = re.compile(activity_regex, re.IGNORECASE) if activity_regex else None
    parts = []

    for f in files:
        act = _activity_from_filename(f)
        if rgx and not rgx.search(act):
            continue
        try:
            parts.append(read_sisfall_txt(f))
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")

    if not parts:
        raise ValueError("No SisFall files parsed")

    return pd.concat(parts, ignore_index=True)