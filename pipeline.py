"""
End-to-end pipeline: load IMU CSV -> process -> extract features -> (optional) ML.
"""

import numpy as np
import pandas as pd
from scipy import interpolate

import config
from signal_processing import apply_lowpass_to_imu, madgwick_fusion, quaternion_to_euler
from features import (
    extract_all_features,
    detect_strokes_z,
)
from visualization import (
    plot_imu_timeseries,
    plot_stroke_metrics,
    plot_z_accel_with_peaks,
)


def _detect_separator(filepath: str) -> str:
    """Peek at the first line to decide if the file is tab- or comma-separated."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if "\t" in header and header.count("\t") > header.count(","):
        return "\t"
    return ","


def _try_column_sets(df, current, *alternatives):
    """Return the first column set whose x/y/z columns all exist in *df*."""
    if current["x"] in df.columns:
        return current
    for alt in alternatives:
        if alt and all(alt[k] in df.columns for k in ("x", "y", "z")):
            return alt
    return current


def load_imu_csv(
    filepath: str,
    time_col: str = None,
    accel_cols: dict = None,
    gyro_cols: dict = None,
    accel_units: str = None,
) -> tuple[pd.DataFrame, float]:
    """
    Load IMU data from CSV (or tab-separated file).
    Auto-detects separator, column names, datetime vs numeric timestamps, and units.
    Returns (dataframe with logical column names, sample_rate_hz).
    """
    sep = _detect_separator(filepath)
    df = pd.read_csv(filepath, sep=sep)
    if len(df) == 0:
        raise ValueError("CSV file is empty or has no data rows. Please upload a file with at least one row of IMU data.")

    time_col = time_col or config.DEFAULT_TIME_COL
    accel_cols = accel_cols or config.DEFAULT_ACCEL_COLS
    gyro_cols = gyro_cols or config.DEFAULT_GYRO_COLS
    accel_units = accel_units or config.ACCEL_UNITS

    device_accel = getattr(config, "DEVICE_ACCEL_COLS", None)
    device_gyro = getattr(config, "DEVICE_GYRO_COLS", None)
    device_time = getattr(config, "DEVICE_TIME_COL", None)
    shimmer_accel = getattr(config, "SHIMMER_ACCEL_COLS", None)
    shimmer_gyro = getattr(config, "SHIMMER_GYRO_COLS", None)
    simple_accel = getattr(config, "SIMPLE_ACCEL_COLS", None)
    simple_gyro = getattr(config, "SIMPLE_GYRO_COLS", None)

    accel_cols = _try_column_sets(df, accel_cols, device_accel, shimmer_accel, simple_accel)
    gyro_cols = _try_column_sets(df, gyro_cols, device_gyro, shimmer_gyro, simple_gyro)

    # Auto-detect accel units from column names (e.g. "AccX(g)" → g)
    detected_units = accel_units
    if device_accel and accel_cols["x"] == device_accel["x"]:
        detected_units = "g"

    # --- Resolve time column ---
    if time_col not in df.columns:
        candidates = [device_time, "timestamp", "Timestamp", "Time", "time", "t"]
        for c in candidates:
            if c and c in df.columns:
                time_col = c
                break
        else:
            df["_gen_time"] = np.arange(len(df)) / config.DEFAULT_SAMPLE_RATE_HZ
            time_col = "_gen_time"

    # Parse time: try numeric first; fall back to datetime strings
    t_raw = df[time_col]
    t_numeric = pd.to_numeric(t_raw, errors="coerce")
    if t_numeric.notna().sum() > 0.5 * len(t_numeric):
        t = t_numeric.ffill().bfill()
        if t.iloc[0] > 1e6:
            t = t / 1000.0
        t = t - t.iloc[0]
    else:
        t_dt = pd.to_datetime(t_raw, errors="coerce")
        if t_dt.notna().sum() < 0.5 * len(t_dt):
            raise ValueError(
                f"Could not parse time column '{time_col}'. "
                "Need either numeric seconds/ms or datetime strings."
            )
        t_dt = t_dt.ffill().bfill()
        t = (t_dt - t_dt.iloc[0]).dt.total_seconds()

    df = df.copy()
    df["time_s"] = t.values

    # Build accel array (Nx3)
    ax = pd.to_numeric(df.get(accel_cols["x"]), errors="coerce").fillna(0).values
    ay = pd.to_numeric(df.get(accel_cols["y"]), errors="coerce").fillna(0).values
    az = pd.to_numeric(df.get(accel_cols["z"]), errors="coerce").fillna(0).values
    accel = np.column_stack([ax, ay, az])
    if detected_units == "g":
        accel = accel * 9.81

    # Build gyro array (Nx3)
    gx = pd.to_numeric(df.get(gyro_cols["x"]), errors="coerce").fillna(0).values
    gy = pd.to_numeric(df.get(gyro_cols["y"]), errors="coerce").fillna(0).values
    gz = pd.to_numeric(df.get(gyro_cols["z"]), errors="coerce").fillna(0).values
    gyro = np.column_stack([gx, gy, gz])
    # Convert deg/s → rad/s when values are clearly in degrees (large magnitudes)
    gyro_max = np.abs(gyro).max()
    if gyro_max > 20:
        gyro = np.radians(gyro)

    # Estimate sample rate from timestamps
    if len(t) > 1:
        dt = np.diff(t.values if hasattr(t, "values") else t)
        dt_pos = dt[dt > 0]
        sample_rate_hz = 1.0 / np.median(dt_pos) if len(dt_pos) else config.DEFAULT_SAMPLE_RATE_HZ
    else:
        sample_rate_hz = config.DEFAULT_SAMPLE_RATE_HZ

    df["accel_x"] = accel[:, 0]
    df["accel_y"] = accel[:, 1]
    df["accel_z"] = accel[:, 2]
    df["gyro_x"] = gyro[:, 0]
    df["gyro_y"] = gyro[:, 1]
    df["gyro_z"] = gyro[:, 2]
    return df, sample_rate_hz


def _resample_to_target(time_s, accel, gyro, native_hz, target_hz):
    """Resample accel (Nx3) and gyro (Nx3) from native_hz up to target_hz using cubic interpolation."""
    t_start, t_end = time_s[0], time_s[-1]
    n_new = int(round((t_end - t_start) * target_hz)) + 1
    t_new = np.linspace(t_start, t_end, n_new)

    accel_new = np.zeros((n_new, 3))
    gyro_new = np.zeros((n_new, 3))
    for ax in range(3):
        accel_new[:, ax] = interpolate.interp1d(
            time_s, accel[:, ax], kind="cubic", fill_value="extrapolate"
        )(t_new)
        gyro_new[:, ax] = interpolate.interp1d(
            time_s, gyro[:, ax], kind="cubic", fill_value="extrapolate"
        )(t_new)
    return t_new, accel_new, gyro_new


def run_pipeline(
    filepath: str = None,
    accel: np.ndarray = None,
    gyro: np.ndarray = None,
    time_s: np.ndarray = None,
    sample_rate_hz: float = None,
    mass_kg: float = None,
    use_fixed_reach: bool = False,
) -> dict:
    """
    Run full pipeline. Either:
      - filepath: path to CSV (uses load_imu_csv), or
      - accel (Nx3), gyro (Nx3), time_s (N,), sample_rate_hz provided directly.
    Returns dict with: df_raw, df_features, sample_rate_hz, peak_indices, figures (list of matplotlib figures).
    """
    if filepath is not None:
        df_raw, sample_rate_hz = load_imu_csv(filepath)
        time_s = df_raw["time_s"].values
        accel = df_raw[["accel_x", "accel_y", "accel_z"]].values
        gyro = df_raw[["gyro_x", "gyro_y", "gyro_z"]].values
    else:
        if accel is None or gyro is None or time_s is None or sample_rate_hz is None:
            raise ValueError("Provide either filepath or (accel, gyro, time_s, sample_rate_hz)")
        df_raw = pd.DataFrame({
            "time_s": time_s,
            "accel_x": accel[:, 0], "accel_y": accel[:, 1], "accel_z": accel[:, 2],
            "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
        })

    # Resample to target rate if the native rate is too low
    target_hz = getattr(config, "TARGET_SAMPLE_RATE_HZ", 100.0)
    if sample_rate_hz < target_hz * 0.9 and len(time_s) >= 4:
        time_s, accel, gyro = _resample_to_target(
            time_s, accel, gyro, sample_rate_hz, target_hz
        )
        sample_rate_hz = target_hz
        df_raw = pd.DataFrame({
            "time_s": time_s,
            "accel_x": accel[:, 0], "accel_y": accel[:, 1], "accel_z": accel[:, 2],
            "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
        })

    # Filter
    accel_f, gyro_f = apply_lowpass_to_imu(
        accel, gyro, sample_rate_hz,
        cutoff_hz=config.LOWPASS_CUTOFF_HZ,
        order=config.LOWPASS_ORDER,
    )
    peak_idx, _ = detect_strokes_z(accel_f[:, 2], sample_rate_hz)

    # Feature extraction
    df_features = extract_all_features(
        accel, gyro, sample_rate_hz,
        mass_kg=mass_kg,
        use_fixed_reach=use_fixed_reach,
    )

    # Figures
    figures = []
    fig1 = plot_imu_timeseries(time_s, accel_f, gyro_f, title="Filtered IMU data")
    figures.append(fig1)
    fig2 = plot_z_accel_with_peaks(time_s, accel_f[:, 2], peak_idx)
    figures.append(fig2)
    if len(df_features) > 0:
        fig3 = plot_stroke_metrics(df_features, title="Per-stroke metrics")
        figures.append(fig3)

    return {
        "df_raw": df_raw,
        "df_features": df_features,
        "sample_rate_hz": sample_rate_hz,
        "peak_indices": peak_idx,
        "figures": figures,
    }
