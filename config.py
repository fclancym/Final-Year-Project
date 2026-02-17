"""
Configuration and constants for the swimming IMU analysis pipeline.
Adjust these to match your Shimmer IMU output and sampling rate.
"""

# Sampling rate of the IMU (Hz). Shimmer often uses 51.2, 128, 256, or 512.
DEFAULT_SAMPLE_RATE_HZ = 128.0

# Low-pass filter cutoff (Hz) to remove high-frequency noise (turbulence, vibration).
# Should be below Nyquist; typical for stroke motion: 5–15 Hz.
LOWPASS_CUTOFF_HZ = 8.0
LOWPASS_ORDER = 4

# Madgwick filter gain (tune for stability vs responsiveness).
MADGWICK_GAIN = 0.1

# Peak detection for stroke count: minimum height (in m/s² or g depending on your units).
# Raw accel is often in g; we can work in m/s² by multiplying by 9.81.
ACCEL_PEAK_MIN_HEIGHT = 2.0   # m/s², adjust after seeing real data
ACCEL_PEAK_MIN_DISTANCE = 0.3  # min seconds between strokes (e.g. 0.3 s ≈ 200 SPM max)

# Zero-velocity update: window (seconds) around "glide" or entry where velocity is forced to zero.
ZUPT_WINDOW_S = 0.05
# Threshold: treat as zero-velocity when magnitude of acceleration is below this (m/s²).
ZUPT_ACCEL_THRESHOLD = 1.5

# Fallback fixed arm length (m) if double integration + ZUPT is too noisy.
DEFAULT_ARM_LENGTH_M = 0.65

# Swimmer mass (kg) for power calculation. Can be overridden per run.
DEFAULT_SWIMMER_MASS_KG = 70.0

# CSV column mapping. Shimmer / your export may use different names.
# Map logical names -> your CSV column names.
DEFAULT_ACCEL_COLS = {"x": "Accelerometer X", "y": "Accelerometer Y", "z": "Accelerometer Z"}
DEFAULT_GYRO_COLS = {"x": "Gyroscope X", "y": "Gyroscope Y", "z": "Gyroscope Z"}
DEFAULT_TIME_COL = "Timestamp"

# Alternative common names (e.g. Shimmer Consensys)
SHIMMER_ACCEL_COLS = {"x": "Low Noise Accelerometer X", "y": "Low Noise Accelerometer Y", "z": "Low Noise Accelerometer Z"}
SHIMMER_GYRO_COLS = {"x": "Gyroscope X", "y": "Gyroscope Y", "z": "Gyroscope Z"}

# Units: set to "g" if your CSV stores acceleration in g, "m/s2" if in m/s².
ACCEL_UNITS = "m/s2"  # or "g"
