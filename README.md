# MPU6050 Digital Shadow Pipeline

I promise I'll do a proper README this weekend, but the important thing is that the main codes are here and the functions are in the mpu_shadow folder.

The python dependencies are in requirements.txt

Basically to fit the model with a dataset of recorded data, we would use fit_shadow_model.py as:

```bash
python3 fit_shadow_model.py \
  --stationary data/stationary.csv \
  --sixpos data/six_position.csv \
  --warmup data/warmup.csv \
  --output configs/shadow_fitted.json
```

For the mobifall dataset, a kaggle dataset I used to test this, first we need to convert the dataset to a single json with

```bash
python3 convert_mobifall.py \
  --mobifall_root path/to/mobifall/dataset \
  --output refactored_dataset.csv \
  --max_trials 10
```

Then train it with:

```bash
python3 fit_shadow_model_mobifall.py \
  --mobifall_root refactored_dataset.csv \
  --max_trials 10 \
  --output configs/shadow_fitted.json
```

After that, to generate data based in a file, we use:

```bash
python3 replay_shadow.py \
  --input data/adl_or_truth.csv \
  --config configs/shadow_fitted.json \
  --output data/adl_shadowed.csv
```

This repository implements the **full Option 1 pipeline** described in the project proposal:

1. **Data acquisition** from an MPU6050 over I2C (`scripts/mpu6050_logger.py`)
2. **Static six-position calibration** and affine triad model estimation (`mpu_shadow/calibration.py`)
3. **Allan deviation analysis** and stochastic noise fitting (`mpu_shadow/allan.py`)
4. **Temperature / warm-up fitting** (`mpu_shadow/temperature.py`)
5. **Replay-based digital shadow simulator** (`mpu_shadow/simulator.py`)
6. **Host streaming over UART** to a target device (`scripts/stream_shadow_uart.py`)
7. **Embedded C virtual IMU backend** (`embedded/virtual_imu.*`)

## Repository Layout

```text
mpu6050_digital_shadow_regenerated/
├── README.md
├── requirements.txt
├── configs/
│   └── example_shadow_config.json
├── docs/
│   └── data_collection_protocol.md
├── scripts/
│   ├── mpu6050_logger.py
│   ├── fit_shadow_model.py
│   ├── replay_shadow.py
│   └── stream_shadow_uart.py
├── mpu_shadow/
│   ├── __init__.py
│   ├── io.py
│   ├── calibration.py
│   ├── allan.py
│   ├── temperature.py
│   ├── simulator.py
│   └── utils.py
└── embedded/
    ├── imu_types.h
    ├── virtual_imu.h
    ├── virtual_imu.c
    └── example_main.c
```

## Expected CSV Format

All scripts use a CSV containing the following columns:

- `timestamp_s`
- `ax_g`, `ay_g`, `az_g`
- `gx_dps`, `gy_dps`, `gz_dps`
- `temp_c`

For six-position calibration, also add:

- `pose_label` with one of `xp`, `xn`, `yp`, `yn`, `zp`, `zn`

## Example End-to-End Workflow

### 1) Log data from the real MPU6050

```bash
python scripts/mpu6050_logger.py --output data/stationary_3h.csv --duration_s 10800 --rate_hz 100
```

### 2) Fit the digital shadow model

```bash
python scripts/fit_shadow_model.py \
  --stationary data/stationary_3h.csv \
  --sixpos data/six_position.csv \
  --warmup data/warmup.csv \
  --output configs/shadow_fitted.json
```

### 3) Replay a motion log with injected MPU6050 imperfections

```bash
python scripts/replay_shadow.py \
  --input data/adl_or_truth.csv \
  --config configs/shadow_fitted.json \
  --output data/adl_shadowed.csv
```

### 4) Stream shadowed samples to the target device

```bash
python scripts/stream_shadow_uart.py \
  --input data/adl_shadowed.csv \
  --serial /dev/ttyUSB0 \
  --baud 115200
```

## Notes

- The calibration stage uses an **affine accelerometer calibration** from six-position data, which is a practical approximation of the triad model.
- The gyroscope deterministic model includes **bias** and optional temperature coefficients. Full gyro scale-factor calibration requires controlled-rate truth data.
- The stochastic model includes white noise, bias random walk, and optional temperature-dependent bias.
- The simulator is **replay-based**: it consumes a motion sequence (real or synthetic) and injects the identified sensor model.
