import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "model/fall_detection_rnn_model_50hz.keras"
SCALER_PATH = "model/scaler_50hz.gz"
MANIFEST_PATH = "manifest_sisfall.csv"
OUTPUT_DIR = "batch_results"

FS = 50.0
WINDOW_SIZE = 200
THRESHOLD = 0.7
TIME_TOL = 0.03

FEATURES = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "ori_azimuth", "ori_pitch", "ori_roll"
]

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# PREPROCESS / INFERENCE
# ============================================================
def prepare_csv(path):
    df = pd.read_csv(path)

    required = ["timestamp_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["timestamp"] = pd.to_timedelta(df["timestamp_s"], unit="s")
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "ax_g": "acc_x", "ay_g": "acc_y", "az_g": "acc_z",
        "gx_dps": "gyro_x", "gy_dps": "gyro_y", "gz_dps": "gyro_z"
    })

    # Match training rate (50 Hz = 20 ms)
    df = df.resample("20ms").mean().interpolate(method="linear", limit_direction="both")

    # Training features
    df["acc_smv"] = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]].values, axis=1)
    df["gyro_smv"] = np.linalg.norm(df[["gyro_x", "gyro_y", "gyro_z"]].values, axis=1)

    # Orientation not present in replay CSVs -> fill with zeros for compatibility
    df["ori_azimuth"] = 0.0
    df["ori_pitch"] = 0.0
    df["ori_roll"] = 0.0

    df["timestamp_s"] = df.index.total_seconds()
    return df.reset_index(drop=True)


def run_model_on_dataframe(df):
    buffer = []
    rows = []

    for _, r in df.iterrows():
        buffer.append(r[FEATURES].values.astype(np.float32))
        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

        prob = np.nan
        detect = 0

        if len(buffer) == WINDOW_SIZE:
            X = scaler.transform(np.array(buffer))
            X = np.expand_dims(X, axis=0)
            pred = model.predict(X, verbose=0)
            prob = float(np.squeeze(pred))
            detect = int(prob > THRESHOLD)

        rows.append({
            "timestamp_s": r["timestamp_s"],
            "fall_probability": prob,
            "fall_detected": detect,
        })

    return pd.DataFrame(rows)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_df(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def align_by_timestamp(df1, df2, suffixes=("_orig", "_shad"), tolerance=0.03):
    df1 = df1.sort_values("timestamp_s").copy()
    df2 = df2.sort_values("timestamp_s").copy()

    merged = pd.merge_asof(
        df1,
        df2,
        on="timestamp_s",
        direction="nearest",
        tolerance=tolerance,
        suffixes=suffixes,
    )
    return merged.dropna().reset_index(drop=True)


def extract_event_starts(df, detection_col="fall_detected"):
    det = df[detection_col].to_numpy(dtype=int)
    t = df["timestamp_s"].to_numpy(dtype=float)

    starts = []
    prev = 0
    for i in range(len(det)):
        if det[i] == 1 and prev == 0:
            starts.append(t[i])
        prev = det[i]
    return starts


def compare_event_latencies(orig_events, shad_events):
    n = min(len(orig_events), len(shad_events))
    if n == 0:
        return []
    return [shad_events[i] - orig_events[i] for i in range(n)]


def load_labels_if_available(labels_csv, reference_df):
    if labels_csv is None or str(labels_csv).strip() == "" or pd.isna(labels_csv):
        return None

    labels_df = pd.read_csv(labels_csv)
    required = {"timestamp_s", "label"}
    if not required.issubset(labels_df.columns):
        raise ValueError(f"Labels file {labels_csv} must contain columns: {required}")

    labels_df = labels_df.sort_values("timestamp_s").copy()
    ref = reference_df[["timestamp_s"]].sort_values("timestamp_s").copy()

    merged = pd.merge_asof(
        ref,
        labels_df,
        on="timestamp_s",
        direction="nearest",
        tolerance=TIME_TOL,
    )

    if merged["label"].isna().all():
        print(f"[WARN] No labels could be aligned from {labels_csv}")
        return None

    return merged["label"].ffill().fillna(0).astype(int).to_numpy()


def false_alarms_per_hour(y_true, y_pred, t):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.nan

    false_events = 0
    prev = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and prev == 0 and y_true[i] == 0:
            false_events += 1
        prev = y_pred[i]

    hours = (t[-1] - t[0]) / 3600.0
    return false_events / hours if hours > 0 else np.nan


def summarize_case_no_labels(case_name, df_orig, df_shad, merged):
    p_orig = merged["fall_probability_orig"].to_numpy(dtype=float)
    p_shad = merged["fall_probability_shad"].to_numpy(dtype=float)
    d_orig = merged["fall_detected_orig"].to_numpy(dtype=int)
    d_shad = merged["fall_detected_shad"].to_numpy(dtype=int)

    orig_events = extract_event_starts(df_orig)
    shad_events = extract_event_starts(df_shad)
    latencies = compare_event_latencies(orig_events, shad_events)

    # Metrics that are valid WITHOUT ground truth
    out = {
        "case_name": case_name,
        "n_samples_aligned": len(merged),

        # Confidence / score consistency
        "orig_mean_probability": float(np.nanmean(p_orig)),
        "orig_max_probability": float(np.nanmax(p_orig)),
        "shad_mean_probability": float(np.nanmean(p_shad)),
        "shad_max_probability": float(np.nanmax(p_shad)),
        "mean_abs_probability_difference": float(np.nanmean(np.abs(p_shad - p_orig))),
        "max_abs_probability_difference": float(np.nanmax(np.abs(p_shad - p_orig))),
        "probability_correlation": float(np.corrcoef(p_orig, p_shad)[0, 1]) if len(p_orig) > 1 else np.nan,

        # Binary decision consistency
        "orig_positive_samples": int(np.sum(d_orig)),
        "shad_positive_samples": int(np.sum(d_shad)),
        "binary_detection_agreement": float(np.mean(d_orig == d_shad)) if len(d_orig) > 0 else np.nan,
        "binary_detection_disagreement": float(np.mean(d_orig != d_shad)) if len(d_orig) > 0 else np.nan,

        # Event consistency
        "orig_detection_events": len(orig_events),
        "shad_detection_events": len(shad_events),
        "event_count_difference": len(shad_events) - len(orig_events),
        "event_latency_mean_s": float(np.mean(latencies)) if latencies else np.nan,
        "event_latency_std_s": float(np.std(latencies)) if latencies else np.nan,
    }
    return out


def summarize_case_with_labels(summary, merged, y_true):
    d_orig = merged["fall_detected_orig"].to_numpy(dtype=int)
    d_shad = merged["fall_detected_shad"].to_numpy(dtype=int)

    summary.update({
        # These are ONLY valid when labels are available
        "orig_accuracy": float(accuracy_score(y_true, d_orig)),
        "orig_precision": float(precision_score(y_true, d_orig, zero_division=0)),
        "orig_recall": float(recall_score(y_true, d_orig, zero_division=0)),
        "orig_f1": float(f1_score(y_true, d_orig, zero_division=0)),
        "orig_false_alarms_per_hour": float(false_alarms_per_hour(y_true, d_orig, merged["timestamp_s"])),

        "shad_accuracy": float(accuracy_score(y_true, d_shad)),
        "shad_precision": float(precision_score(y_true, d_shad, zero_division=0)),
        "shad_recall": float(recall_score(y_true, d_shad, zero_division=0)),
        "shad_f1": float(f1_score(y_true, d_shad, zero_division=0)),
        "shad_false_alarms_per_hour": float(false_alarms_per_hour(y_true, d_shad, merged["timestamp_s"])),
    })
    return summary


def print_case_summary(summary, has_labels=False):
    print("\n" + "=" * 78)
    print(f"CASE: {summary['case_name']}")
    print("=" * 78)

    print("Detector-output consistency metrics (valid without labels):")
    print(f"  Original mean probability:          {summary['orig_mean_probability']:.4f}")
    print(f"  Shadowed mean probability:          {summary['shad_mean_probability']:.4f}")
    print(f"  Original max probability:           {summary['orig_max_probability']:.4f}")
    print(f"  Shadowed max probability:           {summary['shad_max_probability']:.4f}")
    print(f"  Mean |probability difference|:      {summary['mean_abs_probability_difference']:.4f}")
    print(f"  Max  |probability difference|:      {summary['max_abs_probability_difference']:.4f}")
    print(f"  Probability correlation:            {summary['probability_correlation']:.4f}")
    print(f"  Binary detection agreement:         {summary['binary_detection_agreement']:.4f}")
    print(f"  Binary detection disagreement:      {summary['binary_detection_disagreement']:.4f}")
    print(f"  Original positive samples:          {summary['orig_positive_samples']}")
    print(f"  Shadowed positive samples:          {summary['shad_positive_samples']}")
    print(f"  Original detection events:          {summary['orig_detection_events']}")
    print(f"  Shadowed detection events:          {summary['shad_detection_events']}")
    print(f"  Event count difference (shad-orig): {summary['event_count_difference']}")

    if not np.isnan(summary["event_latency_mean_s"]):
        print(f"  Event latency shift mean [s]:       {summary['event_latency_mean_s']:.4f}")
        print(f"  Event latency shift std  [s]:       {summary['event_latency_std_s']:.4f}")
    else:
        print("  Event latency shift:                N/A")

    if has_labels:
        print("\nSupervised task metrics (ONLY valid because labels were provided):")
        print(f"  Original accuracy:                  {summary['orig_accuracy']:.4f}")
        print(f"  Shadowed accuracy:                  {summary['shad_accuracy']:.4f}")
        print(f"  Original precision:                 {summary['orig_precision']:.4f}")
        print(f"  Shadowed precision:                 {summary['shad_precision']:.4f}")
        print(f"  Original recall:                    {summary['orig_recall']:.4f}")
        print(f"  Shadowed recall:                    {summary['shad_recall']:.4f}")
        print(f"  Original F1-score:                  {summary['orig_f1']:.4f}")
        print(f"  Shadowed F1-score:                  {summary['shad_f1']:.4f}")
        print(f"  Original false alarms/hour:         {summary['orig_false_alarms_per_hour']:.4f}")
        print(f"  Shadowed false alarms/hour:         {summary['shad_false_alarms_per_hour']:.4f}")


def plot_case(merged, case_name, out_dir):
    import matplotlib.pyplot as plt

    t = merged["timestamp_s"].to_numpy()
    p_orig = merged["fall_probability_orig"].to_numpy()
    p_shad = merged["fall_probability_shad"].to_numpy()
    d_orig = merged["fall_detected_orig"].to_numpy()
    d_shad = merged["fall_detected_shad"].to_numpy()

    prob_diff = p_shad - p_orig
    det_diff = d_shad - d_orig

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axs[0].plot(t, p_orig, label="Original", color="blue")
    axs[0].plot(t, p_shad, label="Shadowed", color="red")
    axs[0].set_ylabel("Prob.")
    axs[0].set_title(f"{case_name} – Fall probability")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, d_orig, label="Original", color="blue", drawstyle="steps-post")
    axs[1].plot(t, d_shad, label="Shadowed", color="red", drawstyle="steps-post")
    axs[1].set_ylabel("Detect")
    axs[1].set_title("Binary detections")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, prob_diff, color="black")
    axs[2].set_ylabel("Δ prob")
    axs[2].set_title("Probability difference (shadowed - original)")
    axs[2].grid(True)

    axs[3].plot(t, det_diff, color="purple", drawstyle="steps-post")
    axs[3].set_ylabel("Δ det")
    axs[3].set_xlabel("Time [s]")
    axs[3].set_title("Detection difference (shadowed - original)")
    axs[3].grid(True)

    plt.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / f"{case_name}_comparison.png", dpi=200)
    plt.close(fig)

def run_case(case_name, original_csv, shadowed_csv, labels_csv, output_dir):
    print(f"\n>>> Running case: {case_name}")

    case_dir = Path(output_dir) / case_name
    ensure_dir(case_dir)

    # Inference
    df_orig = run_model_on_dataframe(prepare_csv(original_csv))
    df_shad = run_model_on_dataframe(prepare_csv(shadowed_csv))

    save_df(df_orig, case_dir / "original_predictions.csv")
    save_df(df_shad, case_dir / "shadowed_predictions.csv")

    # Alignment
    merged = align_by_timestamp(df_orig, df_shad, tolerance=TIME_TOL)
    save_df(merged, case_dir / "comparison_predictions.csv")

    # Always-valid summary (no labels required)
    summary = summarize_case_no_labels(case_name, df_orig, df_shad, merged)

    # Optional supervised summary
    y_true = load_labels_if_available(labels_csv, merged) if labels_csv else None
    if y_true is not None:
        summary = summarize_case_with_labels(summary, merged, y_true)

    # Output
    print_case_summary(summary, has_labels=(y_true is not None))
    plot_case(merged, case_name, case_dir)

    return summary

def main():
    ensure_dir(OUTPUT_DIR)

    manifest = pd.read_csv(MANIFEST_PATH)
    required_cols = {"case_name", "original_csv", "shadowed_csv"}
    if not required_cols.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns: {required_cols}")

    all_summaries = []
    for _, row in manifest.iterrows():
        case_name = row["case_name"]
        original_csv = row["original_csv"]
        shadowed_csv = row["shadowed_csv"]
        labels_csv = row["labels_csv"] if "labels_csv" in row and pd.notna(row["labels_csv"]) else None

        summary = run_case(
            case_name=case_name,
            original_csv=original_csv,
            shadowed_csv=shadowed_csv,
            labels_csv=labels_csv,
            output_dir=OUTPUT_DIR,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    save_df(summary_df, Path(OUTPUT_DIR) / "summary_metrics.csv")

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)
    print(summary_df)


if __name__ == "__main__":
    main()
