"""
Feature extraction: stroke count, stroke reach, stroke power, hand entry angle, SEI.
Uses filtered/fused signals from signal_processing.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

import config
from signal_processing import (
    apply_lowpass_to_imu,
    madgwick_fusion,
    quaternion_to_euler,
    compute_stroke_reach_from_accel,
)


def detect_strokes_z(
    accel_z: np.ndarray,
    sample_rate_hz: float,
    min_height: float = None,
    min_distance_sec: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect stroke events from Z-axis acceleration peaks.
    Returns (peak_indices, peak_timestamps).
    """
    min_height = min_height if min_height is not None else config.ACCEL_PEAK_MIN_HEIGHT
    min_distance_sec = min_distance_sec if min_distance_sec is not None else config.ACCEL_PEAK_MIN_DISTANCE
    min_distance_samples = max(1, int(min_distance_sec * sample_rate_hz))

    peaks, props = scipy_signal.find_peaks(
        accel_z,
        height=min_height,
        distance=min_distance_samples,
    )
    timestamps = peaks.astype(float) / sample_rate_hz
    return peaks, timestamps


def stroke_rate_from_peaks(peak_timestamps: np.ndarray, window_sec: float = 60.0) -> float:
    """Strokes per minute over a time window (e.g. last 60 s or full session)."""
    if len(peak_timestamps) < 2:
        return 0.0
    t_span = peak_timestamps[-1] - peak_timestamps[0]
    if t_span <= 0:
        return 0.0
    strokes_per_sec = (len(peak_timestamps) - 1) / t_span
    return strokes_per_sec * 60.0  # SPM


def per_stroke_rates(peak_indices: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Stroke rate (SPM) for each stroke, using interval to next stroke."""
    n = len(peak_indices)
    if n < 2:
        return np.array([0.0] * n)
    dt_strokes = np.diff(peak_indices) / sample_rate_hz  # seconds between strokes
    spm = 60.0 / dt_strokes  # SPM for each interval
    return np.concatenate([[spm[0]], spm])  # first stroke gets first interval rate


def hand_entry_angle_at_peaks(
    euler_rad: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """
    Hand entry angle from pitch/roll at stroke (peak) times.
    euler_rad: Nx3 [roll, pitch, yaw] in radians.
    Returns angles in degrees: entry angle per stroke (simplified: use pitch as proxy).
    """
    if len(peak_indices) == 0:
        return np.array([])
    # Use pitch (euler_rad[:, 1]) as proxy for entry angle; convert to degrees.
    angles_rad = euler_rad[peak_indices, 1]
    return np.degrees(angles_rad)


def stroke_reach_per_stroke(
    disp_z: np.ndarray,
    peak_indices: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    Reach per stroke: max displacement in Z between consecutive stroke peaks (or use fixed arm length).
    disp_z: displacement from double integration (or zeros if using fixed L).
    """
    if len(peak_indices) < 2:
        return np.array([config.DEFAULT_ARM_LENGTH_M] * len(peak_indices))
    reaches = []
    for i in range(len(peak_indices)):
        start = peak_indices[i]
        end = peak_indices[i + 1] if i + 1 < len(peak_indices) else len(disp_z)
        segment = disp_z[start:end]
        if len(segment) == 0:
            reaches.append(config.DEFAULT_ARM_LENGTH_M)
        else:
            range_d = np.nanmax(segment) - np.nanmin(segment)
            if range_d < 0.1:  # likely drift/noise
                range_d = config.DEFAULT_ARM_LENGTH_M
            reaches.append(max(range_d, 0.1))
    return np.array(reaches)


def stroke_power_per_stroke(
    accel_z: np.ndarray,
    peak_indices: np.ndarray,
    reach_per_stroke: np.ndarray,
    spm_per_stroke: np.ndarray,
    mass_kg: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    P = F * D * (SR in strokes/sec), with F = m * a (max propulsive accel in pull phase).
    Units: F [N], D [m], SR [1/s] -> P [W].
    """
    n = len(peak_indices)
    if n == 0:
        return np.array([])
    power = np.zeros(n)
    for i in range(n):
        start = peak_indices[i]
        end = peak_indices[i + 1] if i + 1 < n else len(accel_z)
        segment = accel_z[start:end]
        a_max = np.max(segment) if len(segment) else 0.0
        force_n = mass_kg * a_max
        reach = reach_per_stroke[i] if i < len(reach_per_stroke) else config.DEFAULT_ARM_LENGTH_M
        sr_sec = spm_per_stroke[i] / 60.0 if i < len(spm_per_stroke) else 0.0
        power[i] = force_n * reach * sr_sec
    return power


def compute_sei(reach: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Stroke Efficiency Index = Reach / Power. Avoid div by zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sei = np.where(power > 0, reach / power, np.nan)
    return sei


def extract_all_features(
    accel: np.ndarray,
    gyro: np.ndarray,
    sample_rate_hz: float,
    mass_kg: float = None,
    use_fixed_reach: bool = False,
) -> pd.DataFrame:
    """
    Full feature extraction from raw accel (Nx3) and gyro (Nx3).
    Returns a DataFrame with one row per detected stroke: stroke_index, time_s, spm, reach_m, power_w, entry_angle_deg, sei.
    """
    mass_kg = mass_kg or config.DEFAULT_SWIMMER_MASS_KG

    # 1) Low-pass filter
    accel_f, gyro_f = apply_lowpass_to_imu(
        accel, gyro, sample_rate_hz,
        cutoff_hz=config.LOWPASS_CUTOFF_HZ,
        order=config.LOWPASS_ORDER,
    )
    accel_z = accel_f[:, 2]

    # 2) Madgwick fusion -> orientation
    quats = madgwick_fusion(accel_f, gyro_f, sample_rate_hz, gain=config.MADGWICK_GAIN)
    euler_rad = quaternion_to_euler(quats)

    # 3) Stroke detection (peaks on Z)
    peak_idx, peak_ts = detect_strokes_z(accel_z, sample_rate_hz)
    if len(peak_idx) == 0:
        return pd.DataFrame(columns=["stroke_index", "time_s", "spm", "reach_m", "power_w", "entry_angle_deg", "sei"])

    # 4) Stroke rate per stroke
    spm_per_stroke = per_stroke_rates(peak_idx, sample_rate_hz)

    # 5) Reach: dynamic (ZUPT) or fixed
    if use_fixed_reach:
        reach_per_stroke = np.full(len(peak_idx), config.DEFAULT_ARM_LENGTH_M)
    else:
        disp_z = compute_stroke_reach_from_accel(
            accel_z, sample_rate_hz,
            zupt_threshold=config.ZUPT_ACCEL_THRESHOLD,
            zupt_window_sec=config.ZUPT_WINDOW_S,
        )
        reach_per_stroke = stroke_reach_per_stroke(disp_z, peak_idx, sample_rate_hz)

    # 6) Hand entry angle at each peak
    entry_angles = hand_entry_angle_at_peaks(euler_rad, peak_idx)

    # 7) Power per stroke
    power_per_stroke = stroke_power_per_stroke(
        accel_z, peak_idx, reach_per_stroke, spm_per_stroke, mass_kg, sample_rate_hz
    )

    # 8) SEI
    sei = compute_sei(reach_per_stroke, power_per_stroke)

    return pd.DataFrame({
        "stroke_index": np.arange(len(peak_idx)),
        "time_s": peak_ts,
        "spm": spm_per_stroke,
        "reach_m": reach_per_stroke,
        "power_w": power_per_stroke,
        "entry_angle_deg": entry_angles,
        "sei": sei,
    })
