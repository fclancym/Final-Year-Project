"""
Generate synthetic accelerometer and gyroscope data for testing the pipeline
before real IMU data is available. Simulates periodic stroke-like peaks and orientation change.
"""

import numpy as np


def generate_synthetic_imu(
    duration_sec: float = 60.0,
    sample_rate_hz: float = 128.0,
    stroke_rate_spm: float = 50.0,
    noise_level_accel: float = 0.5,
    noise_level_gyro: float = 0.2,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic accel (Nx3) and gyro (Nx3) in m/s² and rad/s, and time vector (N,).
    Simulates periodic propulsive peaks on Z and small orientation changes.
    """
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_sec * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz

    # Stroke period in seconds
    stroke_period = 60.0 / stroke_rate_spm
    # Peaks at stroke times (propulsive phase)
    peak_times = np.arange(0, duration_sec, stroke_period)
    accel_z = np.zeros(n)
    for pt in peak_times:
        idx = int(pt * sample_rate_hz)
        if idx >= n:
            break
        # Gaussian-like impulse (propulsive peak)
        width = int(0.1 * sample_rate_hz)
        for d in range(-width, width + 1):
            i = idx + d
            if 0 <= i < n:
                accel_z[i] += 4.0 * np.exp(-(d ** 2) / (2 * (width / 3) ** 2))

    # Add gravity-like baseline and noise
    accel = np.zeros((n, 3))
    accel[:, 0] = np.random.randn(n) * noise_level_accel
    accel[:, 1] = np.random.randn(n) * noise_level_accel
    accel[:, 2] = accel_z + 2.0 + np.random.randn(n) * noise_level_accel  # forward + gravity component

    # Gyro: small periodic rotation (pitch/roll change at entry)
    phase = 2 * np.pi * t / stroke_period
    gyro = np.zeros((n, 3))
    gyro[:, 0] = 0.5 * np.sin(phase) + np.random.randn(n) * noise_level_gyro
    gyro[:, 1] = 0.3 * np.cos(phase) + np.random.randn(n) * noise_level_gyro
    gyro[:, 2] = np.random.randn(n) * noise_level_gyro
    # Convert to rad/s (already in rad/s if we treat as small angular velocity)
    # No conversion needed if we assume synthetic is in rad/s

    return t, accel, gyro


def generate_synthetic_csv(
    filepath: str,
    duration_sec: float = 60.0,
    sample_rate_hz: float = 128.0,
    stroke_rate_spm: float = 50.0,
    **kwargs,
) -> None:
    """
    Write synthetic IMU data to a CSV with column names matching config.DEFAULT_ACCEL_COLS / GYRO_COLS.
    """
    import pandas as pd
    import config

    t, accel, gyro = generate_synthetic_imu(
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        stroke_rate_spm=stroke_rate_spm,
        **kwargs,
    )
    time_col = config.DEFAULT_TIME_COL
    acc_cols = list(config.DEFAULT_ACCEL_COLS.values())
    gyr_cols = list(config.DEFAULT_GYRO_COLS.values())
    df = pd.DataFrame(
        np.hstack([t.reshape(-1, 1), accel, gyro]),
        columns=[time_col] + acc_cols + gyr_cols,
    )
    df.to_csv(filepath, index=False)
    print(f"Wrote synthetic data to {filepath} ({len(df)} rows)")


if __name__ == "__main__":
    generate_synthetic_csv("synthetic_imu_sample.csv", duration_sec=30, stroke_rate_spm=55)
