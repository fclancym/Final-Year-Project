"""
Visualization: time-series plots for raw/filtered IMU data and extracted metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_imu_timeseries(
    time_s: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    accel_labels: tuple = ("Accel X", "Accel Y", "Accel Z"),
    gyro_labels: tuple = ("Gyro X", "Gyro Y", "Gyro Z"),
    title: str = "IMU raw/filtered data",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot 3-axis accelerometer and gyroscope vs time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    for i, label in enumerate(accel_labels):
        ax1.plot(time_s, accel[:, i], label=label, alpha=0.8)
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    for i, label in enumerate(gyro_labels):
        ax2.plot(time_s, np.degrees(gyro[:, i]), label=label, alpha=0.8)
    ax2.set_ylabel("Angular velocity (°/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_stroke_metrics(df: pd.DataFrame, title: str = "Stroke metrics") -> plt.Figure:
    """
    Plot per-stroke metrics: SPM, reach, power, entry angle, SEI.
    df must have columns: time_s, spm, reach_m, power_w, entry_angle_deg, sei.
    """
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(14, 8))
    t = df["time_s"].values

    axes[0, 0].plot(t, df["spm"], "b.-", markersize=4)
    axes[0, 0].set_ylabel("Stroke rate (SPM)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, df["reach_m"], "g.-", markersize=4)
    axes[0, 1].set_ylabel("Reach (m)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(t, df["power_w"], "r.-", markersize=4)
    axes[0, 2].set_ylabel("Power (W)")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(t, df["entry_angle_deg"], "m.-", markersize=4)
    axes[1, 0].set_ylabel("Entry angle (°)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    sei = df["sei"].replace([np.inf, -np.inf], np.nan)
    axes[1, 1].plot(t, sei, "c.-", markersize=4)
    axes[1, 1].set_ylabel("SEI (Reach/Power)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    # Summary text
    summary = (
        f"Strokes: {len(df)}\n"
        f"Mean SPM: {df['spm'].mean():.1f}\n"
        f"Mean reach: {df['reach_m'].mean():.2f} m\n"
        f"Mean power: {df['power_w'].mean():.1f} W\n"
        f"Mean SEI: {sei.mean():.4f}"
    )
    axes[1, 2].text(0.1, 0.5, summary, transform=axes[1, 2].transAxes, fontsize=11, verticalalignment="center")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_z_accel_with_peaks(
    time_s: np.ndarray,
    accel_z: np.ndarray,
    peak_indices: np.ndarray,
    title: str = "Z-axis acceleration and detected strokes",
) -> plt.Figure:
    """Plot Z acceleration and mark detected stroke peaks."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_s, accel_z, "b-", alpha=0.7, label="Accel Z")
    if len(peak_indices) > 0:
        peak_t = time_s[peak_indices]
        peak_a = accel_z[peak_indices]
        ax.scatter(peak_t, peak_a, color="red", s=30, zorder=5, label=f"Strokes ({len(peak_indices)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sei_with_change_points(
    df: pd.DataFrame,
    change_point_stroke_indices: list = None,
    title: str = "Stroke Efficiency Index and change points",
) -> plt.Figure:
    """Plot SEI time series and optional change points (from ruptures)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    t = df["time_s"].values
    sei = df["sei"].replace([np.inf, -np.inf], np.nan)
    ax.plot(t, sei, "b.-", markersize=4, label="SEI")
    if change_point_stroke_indices:
        for idx in change_point_stroke_indices:
            if 0 <= idx < len(t):
                ax.axvline(t[idx], color="red", linestyle="--", alpha=0.7)
        ax.legend(["SEI", "Change point"])
    else:
        ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SEI (Reach / Power)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
