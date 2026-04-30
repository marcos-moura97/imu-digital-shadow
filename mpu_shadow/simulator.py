from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .utils import ensure_array

@dataclass
class SensorParams:
    A: list
    bias: list
    noise_density: list
    bias_random_walk: list
    temp_coeff: list
    temp_ref_c: float

@dataclass
class ShadowConfig:
    sample_rate_hz: float
    accelerometer: SensorParams
    gyroscope: SensorParams

    @staticmethod
    def from_dict(d):
        return ShadowConfig(
            sample_rate_hz=float(d['sample_rate_hz']),
            accelerometer=SensorParams(**d['accelerometer']),
            gyroscope=SensorParams(**d['gyroscope']),
        )

    def to_dict(self):
        return asdict(self)

class ShadowIMU:
    def __init__(self, config: ShadowConfig, rng: np.random.Generator | None = None):
        self.cfg = config
        self.dt = 1.0 / config.sample_rate_hz
        self.rng = rng or np.random.default_rng()

        self.acc_bias_rw = np.zeros(3)
        self.gyr_bias_rw = np.zeros(3)

        self.acc_A = ensure_array(config.accelerometer.A, (3, 3))
        self.acc_bias = ensure_array(config.accelerometer.bias, (3,))
        self.acc_noise_density = ensure_array(config.accelerometer.noise_density, (3,))
        self.acc_brw = ensure_array(config.accelerometer.bias_random_walk, (3,))
        self.acc_temp_coeff = ensure_array(config.accelerometer.temp_coeff, (3,))
        self.acc_tref = float(config.accelerometer.temp_ref_c)

        self.gyr_A = ensure_array(config.gyroscope.A, (3, 3))
        self.gyr_bias = ensure_array(config.gyroscope.bias, (3,))
        self.gyr_noise_density = ensure_array(config.gyroscope.noise_density, (3,))
        self.gyr_brw = ensure_array(config.gyroscope.bias_random_walk, (3,))
        self.gyr_temp_coeff = ensure_array(config.gyroscope.temp_coeff, (3,))
        self.gyr_tref = float(config.gyroscope.temp_ref_c)

    def reset(self):
        self.acc_bias_rw[:] = 0.0
        self.gyr_bias_rw[:] = 0.0

    def _step_bias(self):
        self.acc_bias_rw += self.rng.normal(0.0, self.acc_brw * np.sqrt(self.dt), size=3)
        self.gyr_bias_rw += self.rng.normal(0.0, self.gyr_brw * np.sqrt(self.dt), size=3)

    def simulate_sample(self, acc_true_g, gyr_true_dps, temp_c: float):
        acc_true_g = ensure_array(acc_true_g, (3,))
        gyr_true_dps = ensure_array(gyr_true_dps, (3,))
        self._step_bias()

        acc_noise = self.rng.normal(0.0, self.acc_noise_density / np.sqrt(self.dt), size=3)
        gyr_noise = self.rng.normal(0.0, self.gyr_noise_density / np.sqrt(self.dt), size=3)

        acc_temp = self.acc_temp_coeff * (temp_c - self.acc_tref)
        gyr_temp = self.gyr_temp_coeff * (temp_c - self.gyr_tref)

        acc_meas = self.acc_A @ acc_true_g + self.acc_bias + self.acc_bias_rw + acc_temp + acc_noise
        gyr_meas = self.gyr_A @ gyr_true_dps + self.gyr_bias + self.gyr_bias_rw + gyr_temp + gyr_noise
        return acc_meas, gyr_meas

    def simulate_dataframe(self, df_truth: pd.DataFrame) -> pd.DataFrame:
        out = df_truth.copy()
        acc_cols = ["ax_g", "ay_g", "az_g"]
        gyr_cols = ["gx_dps", "gy_dps", "gz_dps"]
        acc_out, gyr_out = [], []
        for _, row in df_truth.iterrows():
            acc_true = row[acc_cols].to_numpy(dtype=float)
            gyr_true = row[gyr_cols].to_numpy(dtype=float)
            temp_c = float(row.get('temp_c', self.acc_tref))
            acc_meas, gyr_meas = self.simulate_sample(acc_true, gyr_true, temp_c)
            acc_out.append(acc_meas)
            gyr_out.append(gyr_meas)
        out[acc_cols] = np.vstack(acc_out)
        out[gyr_cols] = np.vstack(gyr_out)
        return out
