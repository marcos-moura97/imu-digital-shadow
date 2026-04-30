from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# General notes
# ---------------------------------------------------------------------
# This adapter is intended for raw MobiFall-style folder trees where each
# trial contains separate .txt files for:
#   - accelerometer
#   - gyroscope
#   - orientation (optional, ignored here)
#
# It outputs a unified dataframe with the internal schema expected by the
# digital-shadow pipeline:
#
#   timestamp_s,
#   ax_g, ay_g, az_g,
#   gx_dps, gy_dps, gz_dps,
#   temp_c
#
# Since MobiFall does not provide temperature measurements, temp_c is set
# to a constant placeholder (25.0).
#
# The parser is intentionally robust / best-effort, because mirrors of the
# dataset may differ slightly in filename patterns and header formatting.
# ---------------------------------------------------------------------


NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def split_by_trial(df: pd.DataFrame):
    """
    Yield per-trial dataframes using the 'trial_key' column.
    """
    if "trial_key" not in df.columns:
        raise ValueError("MobiFall dataframe must contain 'trial_key'")
    for _, sub in df.groupby("trial_key", sort=False):
        yield sub.reset_index(drop=True)


def quasistatic_mask(
    df: pd.DataFrame,
    gyro_threshold_dps: float = 5.0,
    acc_mag_tol_g: float = 0.10,
):
    """
    Return a boolean mask for quasi-static samples.
    """
    gmag = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
    wmag = np.sqrt(df["gx_dps"]**2 + df["gy_dps"]**2 + df["gz_dps"]**2)

    return ((np.abs(gmag - 1.0) < acc_mag_tol_g) & (wmag < gyro_threshold_dps)).to_numpy()


def contiguous_runs(mask: np.ndarray, min_run_length: int = 200):
    """
    Return list of (start, end) indices for contiguous True runs
    of length >= min_run_length.
    """
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_run_length:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_run_length:
        runs.append((start, len(mask)))
    return runs

def _parse_numeric_line(line: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse one raw sensor line from a MobiFall .txt file.
    The function extracts all numeric tokens and returns the last 4 as:
        (timestamp_raw, x, y, z)
    """
    nums = [float(x) for x in NUM_RE.findall(line)]
    if len(nums) < 4:
        return None
    vals = nums[-4:]
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def read_mobifall_sensor_txt(path: str | Path) -> pd.DataFrame:
    """
    Read one raw MobiFall sensor text file (accelerometer or gyroscope)
    """
    path = Path(path)
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_numeric_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise ValueError(f"No numeric sensor rows found in {path}")

    return pd.DataFrame(rows, columns=["timestamp_raw", "x", "y", "z"])


def _detect_sensor_kind(path: Path) -> Optional[str]:
    """
    Guess sensor kind from filename.
    """
    s = path.stem.lower()

    if "acc" in s:
        return "acc"
    if "gyr" in s or "gyro" in s:
        return "gyro"
    if "ori" in s or "orientation" in s:
        return "ori"

    return None


def _normalize_trial_key(path: Path) -> str:
    """
    Build a normalized grouping key so files belonging to the same trial
    can be paired together.
    """
    s = path.stem.lower()

    for token in [
        "accelerometer",
        "acceleration",
        "accel",
        "acc",
        "gyroscope",
        "gyro",
        "gyr",
        "orientation",
        "ori",
    ]:
        s = s.replace(token, "")

    s = re.sub(r"[_\-\s]+", "_", s).strip("_")
    return f"{path.parent.as_posix()}::{s}"


def discover_mobifall_trials(root: str | Path) -> List[Dict[str, Path]]:
    """
    Discover raw MobiFall trials from a root folder.

    The function recursively scans for .txt files, groups them by trial,
    and returns only groups containing at least accel or gyro
    """
    root = Path(root)
    groups: Dict[str, Dict[str, Path]] = {}

    for p in root.rglob("*.txt"):
        kind = _detect_sensor_kind(p)
        if kind is None:
            continue

        key = _normalize_trial_key(p)
        groups.setdefault(key, {})[kind] = p

    trials = []
    for key, files in groups.items():
        if "acc" in files and "gyro" in files:
            trials.append({"key": key, **files})

    return sorted(trials, key=lambda d: d["key"])


def _convert_acc_to_g(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert accelerometer values from m/s^2 to g.
    """
    out = df.copy()
    out[["x", "y", "z"]] = out[["x", "y", "z"]] / 9.80665
    return out


def _convert_gyro_to_dps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert gyroscope values from rad/s to deg/s.
    """
    out = df.copy()
    out[["x", "y", "z"]] = np.rad2deg(out[["x", "y", "z"]])
    return out


def _subject_from_path(path: Path) -> str:
    """
    Try to infer subject ID from the folder path, e.g. sub1, sub12, ...
    """
    for part in path.parts:
        if part.lower().startswith("sub"):
            return part
    return "unknown"


def _activity_from_path(path: Path) -> str:
    """
    Infer activity code from path.

    Usually MobiFall mirrors place trials in folders like:
        sub1/FALLS/
        sub2/STD/
        sub3/WAL/
    """
    parent = path.parent.name
    return parent if parent else path.stem.split("_")[0]


def merge_mobifall_trial(
    acc_path: Path,
    gyro_path: Path,
    add_metadata: bool = True,
) -> pd.DataFrame:
    """
    Merge one accelerometer file + one gyroscope file into the internal schema.
    """
    acc = _convert_acc_to_g(read_mobifall_sensor_txt(acc_path))
    gyr = _convert_gyro_to_dps(read_mobifall_sensor_txt(gyro_path))

    t_acc = acc["timestamp_raw"].to_numpy(dtype=float)
    t_gyr = gyr["timestamp_raw"].to_numpy(dtype=float)

    if len(t_acc) == 0 or len(t_gyr) == 0:
        raise ValueError(f"Empty sensor stream in trial: {acc_path}, {gyro_path}")

    # Use gyroscope timestamps as the reference timeline
    t = t_gyr

    ax = np.interp(t, t_acc, acc["x"])
    ay = np.interp(t, t_acc, acc["y"])
    az = np.interp(t, t_acc, acc["z"])

    gx = np.interp(t, t_gyr, gyr["x"])
    gy = np.interp(t, t_gyr, gyr["y"])
    gz = np.interp(t, t_gyr, gyr["z"])

    t0 = float(t[0])
    ts_s = (t - t0) * 1e-9  # ns -> s relative time

    out = pd.DataFrame(
        {
            "timestamp_s": ts_s,
            "ax_g": ax,
            "ay_g": ay,
            "az_g": az,
            "gx_dps": gx,
            "gy_dps": gy,
            "gz_dps": gz,
            "temp_c": np.full_like(ts_s, 25.0, dtype=float), 
        }
    )

    if add_metadata:
        out["subject_id"] = _subject_from_path(acc_path)
        out["activity_code"] = _activity_from_path(acc_path)
        out["trial_key"] = _normalize_trial_key(acc_path)

    return out


def build_mobifall_dataframe(
    root: str | Path,
    max_trials: Optional[int] = None,
    activity_regex: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a unified dataframe from the entire raw MobiFall root folder.
    """
    trials = discover_mobifall_trials(root)

    if max_trials is not None:
        trials = trials[:max_trials]

    parts = []
    rgx = re.compile(activity_regex, re.IGNORECASE) if activity_regex else None

    for tr in trials:
        activity = _activity_from_path(tr["acc"])
        if rgx and not rgx.search(activity):
            continue

        df = merge_mobifall_trial(tr["acc"], tr["gyro"])
        parts.append(df)

    if not parts:
        raise ValueError("No MobiFall trials matched the requested filters")

    return pd.concat(parts, ignore_index=True)


def infer_sample_rate_hz(df: pd.DataFrame) -> float:
    """
    Infer sampling rate from unified dataframe timestamps.

    Uses the median positive delta-t for robustness.
    """
    dt = df["timestamp_s"].diff().dropna()
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Could not infer sample rate from timestamps")

    return float(1.0 / dt.median())


def standing_like_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a quasi-static subset for benchmark fitting.
    """
    if "activity_code" not in df.columns:
        return df.iloc[0:0].copy()

    mask = df["activity_code"].astype(str).str.contains(r"STD|stand", case=False, regex=True)
    return df[mask].copy()
