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
