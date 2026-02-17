"""
Signal processing for swimming IMU data: filtering, sensor fusion, and ZUPT.
Expects accelerometer in m/s² and gyroscope in rad/s (convert in loader if needed).
"""

import numpy as np
from scipy import signal as scipy_signal


def lowpass_filter(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    Butterworth low-pass filter. Attenuates high-frequency noise.
    """
    nyq = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = scipy_signal.butter(order, normal_cutoff, btype="low", analog=False)
    return scipy_signal.filtfilt(b, a, x, axis=0)


def apply_lowpass_to_imu(
    accel: np.ndarray,
    gyro: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = 8.0,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply low-pass filter to 3-axis accel (Nx3) and gyro (Nx3)."""
    accel_f = lowpass_filter(accel, sample_rate_hz, cutoff_hz, order)
    gyro_f = lowpass_filter(gyro, sample_rate_hz, cutoff_hz, order)
    return accel_f, gyro_f


def _accel_to_quat(a: np.ndarray) -> np.ndarray:
    """Initial quaternion from accelerometer (gravity) direction. Returns [w,x,y,z]."""
    anorm = np.linalg.norm(a)
    if anorm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = a[0] / anorm, a[1] / anorm, a[2] / anorm
    # Pitch and roll from accel; yaw = 0
    pitch = np.arcsin(np.clip(-ax, -1, 1))
    roll = np.arctan2(ay, az)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    w = cr * cp
    x = sr * cp
    y = cr * sp
    z = -sr * sp
    return np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])


def _madgwick_imu_step(q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float, beta: float) -> np.ndarray:
    """One Madgwick IMU update. q, gyr, acc 1d arrays; returns new quat [w,x,y,z]."""
    q = q / np.linalg.norm(q)
    gx, gy, gz = gyr[0], gyr[1], gyr[2]
    # Quaternion rate from gyro: q_dot = 0.5 * q \otimes [0, gyr]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    q_dot_omega = 0.5 * np.array([
        -qx * gx - qy * gy - qz * gz,
        qw * gx + qy * gz - qz * gy,
        qw * gy - qx * gz + qz * gx,
        qw * gz + qx * qy - qy * gx,
    ])
    anorm = np.linalg.norm(acc)
    if anorm > 1e-9:
        ax, ay, az = acc[0] / anorm, acc[1] / anorm, acc[2] / anorm
        f = np.array([
            2.0 * (qx * qz - qw * qy) - ax,
            2.0 * (qw * qx + qy * qz) - ay,
            2.0 * (0.5 - qx ** 2 - qy ** 2) - az,
        ])
        J = np.array([
            [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
            [2 * qx, 2 * qw, 2 * qz, 2 * qy],
            [0.0, -4 * qx, -4 * qy, 0.0],
        ])
        grad = J.T @ f
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-9:
            q_dot_omega -= beta * (grad / gnorm)
    q_new = q + q_dot_omega * dt
    return q_new / np.linalg.norm(q_new)


def madgwick_fusion(
    accel: np.ndarray,
    gyro: np.ndarray,
    sample_rate_hz: float,
    gain: float = 0.1,
) -> np.ndarray:
    """
    Fuse accel (Nx3 m/s²) and gyro (Nx3 rad/s) with Madgwick filter.
    Returns quaternions (Nx4) [w, x, y, z]. Uses a self-contained implementation
    so behaviour is consistent across environments (no ahrs constructor issues).
    """
    n = len(accel)
    dt = 1.0 / sample_rate_hz
    quats = np.zeros((n, 4))
    quats[0] = _accel_to_quat(accel[0])
    for t in range(1, n):
        quats[t] = _madgwick_imu_step(quats[t - 1], gyro[t], accel[t], dt, gain)
    return quats


def quaternion_to_euler(quats: np.ndarray) -> np.ndarray:
    """
    Convert quaternions (Nx4) [w,x,y,z] to Euler angles (Nx3) [roll, pitch, yaw] in radians.
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # Roll (x), Pitch (y), Yaw (z) - common convention
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def zupt_velocity(
    accel: np.ndarray,
    sample_rate_hz: float,
    threshold: float = 1.5,
    window_sec: float = 0.05,
) -> np.ndarray:
    """
    Integrate acceleration to velocity, applying Zero Velocity Updates where
    acceleration magnitude is below threshold. Returns velocity (Nx3) in m/s.
    """
    dt = 1.0 / sample_rate_hz
    n = len(accel)
    vel = np.zeros_like(accel)
    mag = np.linalg.norm(accel, axis=1)
    window_samples = max(1, int(window_sec * sample_rate_hz))

    for i in range(1, n):
        vel[i] = vel[i - 1] + accel[i] * dt
        # ZUPT: in a window around current sample, if accel is near 0, reset velocity
        start = max(0, i - window_samples)
        end = min(n, i + window_samples + 1)
        if np.any(mag[start:end] < threshold):
            vel[i] *= 0.0  # force to zero at rest

    return vel


def velocity_to_displacement_zupt(
    vel: np.ndarray,
    sample_rate_hz: float,
    threshold_vel: float = 0.1,
) -> np.ndarray:
    """
    Integrate velocity to displacement. Optionally zero out displacement when
    velocity is below threshold (drift reset). Returns displacement (Nx3) in m.
    """
    dt = 1.0 / sample_rate_hz
    disp = np.zeros_like(vel)
    vel_mag = np.linalg.norm(vel, axis=1)

    for i in range(1, len(vel)):
        disp[i] = disp[i - 1] + vel[i] * dt
        if vel_mag[i] < threshold_vel:
            disp[i] = disp[i] * 0.0  # optional: reset at rest

    return disp


def compute_stroke_reach_from_accel(
    accel_z: np.ndarray,
    sample_rate_hz: float,
    zupt_threshold: float = 1.5,
    zupt_window_sec: float = 0.05,
) -> np.ndarray:
    """
    One-dimensional stroke reach from Z-axis acceleration only (simplified).
    Double integration with ZUPT on velocity. Returns displacement (N,) in m.
    """
    accel_1d = np.zeros((len(accel_z), 3))
    accel_1d[:, 2] = accel_z
    vel = zupt_velocity(accel_1d, sample_rate_hz, zupt_threshold, zupt_window_sec)
    vel_z = vel[:, 2]
    # Second integration: displacement (with simple drift reset at near-zero velocity)
    dt = 1.0 / sample_rate_hz
    disp_z = np.zeros(len(vel_z))
    for i in range(1, len(vel_z)):
        disp_z[i] = disp_z[i - 1] + vel_z[i] * dt
        if abs(vel_z[i]) < 0.1:
            disp_z[i] = 0.0
    return disp_z
