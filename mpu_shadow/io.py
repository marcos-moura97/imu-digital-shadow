from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "temp_c"
]


def load_imu_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def save_imu_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
