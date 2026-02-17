"""
End-to-end pipeline: load IMU CSV -> process -> extract features -> (optional) ML.
"""

import numpy as np
import pandas as pd

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


def load_imu_csv(
    filepath: str,
    time_col: str = None,
    accel_cols: dict = None,
    gyro_cols: dict = None,
    accel_units: str = None,
) -> tuple[pd.DataFrame, float]:
    """
    Load IMU data from CSV. Returns (dataframe with columns aligned to logical names, sample_rate_hz).
    Assumes timestamp is in seconds or will be converted to elapsed seconds.
    If CSV has no sample rate info, use config.DEFAULT_SAMPLE_RATE_HZ.
    """
    df = pd.read_csv(filepath)
    time_col = time_col or config.DEFAULT_TIME_COL
    accel_cols = accel_cols or config.DEFAULT_ACCEL_COLS
    gyro_cols = gyro_cols or config.DEFAULT_GYRO_COLS
    accel_units = accel_units or config.ACCEL_UNITS

    # Normalize time to seconds from start
    if time_col not in df.columns:
        # Try common alternatives
        for c in ["timestamp", "Timestamp", "Time", "time", "t"]:
            if c in df.columns:
                time_col = c
                break
        else:
            df["Timestamp"] = np.arange(len(df)) / config.DEFAULT_SAMPLE_RATE_HZ
            time_col = "Timestamp"
    t = pd.to_numeric(df[time_col], errors="coerce").ffill()
    if t.iloc[0] > 1e6:  # possibly milliseconds
        t = t / 1000.0
    t = t - t.iloc[0]
    df = df.copy()
    df["time_s"] = t

    # Build accel/gyro arrays (Nx3)
    ax = df[accel_cols["x"]].values if accel_cols["x"] in df.columns else np.zeros(len(df))
    ay = df[accel_cols["y"]].values if accel_cols["y"] in df.columns else np.zeros(len(df))
    az = df[accel_cols["z"]].values if accel_cols["z"] in df.columns else np.zeros(len(df))
    accel = np.column_stack([ax, ay, az])
    if accel_units == "g":
        accel = accel * 9.81

    gx = df[gyro_cols["x"]].values if gyro_cols["x"] in df.columns else np.zeros(len(df))
    gy = df[gyro_cols["y"]].values if gyro_cols["y"] in df.columns else np.zeros(len(df))
    gz = df[gyro_cols["z"]].values if gyro_cols["z"] in df.columns else np.zeros(len(df))
    gyro = np.column_stack([gx, gy, gz])
    # If gyro is in deg/s, convert to rad/s
    if np.abs(gyro).max() < 20 and np.abs(gyro).max() > 0:
        gyro = np.radians(gyro)

    # Estimate sample rate from timestamps if possible
    if len(t) > 1:
        dt = np.diff(t)
        sample_rate_hz = 1.0 / np.median(dt[dt > 0])
    else:
        sample_rate_hz = config.DEFAULT_SAMPLE_RATE_HZ

    df["accel_x"] = accel[:, 0]
    df["accel_y"] = accel[:, 1]
    df["accel_z"] = accel[:, 2]
    df["gyro_x"] = gyro[:, 0]
    df["gyro_y"] = gyro[:, 1]
    df["gyro_z"] = gyro[:, 2]
    return df, sample_rate_hz


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
