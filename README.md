# Final-Year-Project

ML Analytics for Stroke Count, Technique and Fatigue Detection in Swimmers.

## Backend overview

- **Input:** XYZ data from accelerometer and gyroscope (e.g. Shimmer IMU), as CSV or arrays.
- **Processing:** Low-pass filter, Madgwick sensor fusion, ZUPT for drift correction.
- **Features:** Stroke count/rate, stroke reach, stroke power, hand entry angle, Stroke Efficiency Index (SEI).
- **Output:** Per-stroke metrics DataFrame, graphs (accel/gyro time series, power, reach, SEI, etc.), and optional ML (change-point detection, fatigue clustering).

## Running in Google Colab

1. Upload this repo to Colab (e.g. zip the project and upload, or clone from Git).
2. Open `FYP_Swimming_IMU_Pipeline.ipynb`.
3. Run the cells: install deps, set path, run pipeline on synthetic data, view graphs.
4. When you have real data, upload your CSV and run the pipeline with `run_pipeline(filepath='...')`.

## Web dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a local web app at `http://localhost:8501` with:
- **Synthetic data** by default (or upload your CSV)
- **Weight input** for accurate power calculation
- **Black / red / white** theme
- EO-style charts: Stroke Rate & Power, Entry Angle, SEI

## Local / command line

```bash
pip install -r requirements.txt
# Synthetic data (no CSV):
python run_pipeline.py
# Your CSV:
python run_pipeline.py --csv path/to/imu_export.csv --mass 75 --fixed-reach
python run_pipeline.py --help
```

## Project layout

| File | Purpose |
|------|--------|
| `config.py` | Sample rate, filter cutoff, column names, ZUPT and peak-detection settings. |
| `signal_processing.py` | Low-pass filter, Madgwick fusion, ZUPT, double integration for reach. |
| `features.py` | Stroke detection (peaks), reach, power, entry angle, SEI. |
| `visualization.py` | Plots for IMU traces and per-stroke metrics. |
| `pipeline.py` | Load CSV, run processing and features, produce figures. |
| `synthetic_data.py` | Generate fake accel/gyro for testing without real data. |
| `ml_analysis.py` | Change-point detection (ruptures) and K-Means fatigue clustering. |
| `run_pipeline.py` | CLI entry point. |
| `app.py` | Streamlit web dashboard. |
| `FYP_Swimming_IMU_Pipeline.ipynb` | Colab notebook. |

## CSV format

Default expected column names (edit `config.py` or pass to `load_imu_csv` if your export differs):

- **Time:** `Timestamp` (seconds from start, or converted from ms).
- **Accelerometer:** `Accelerometer X`, `Accelerometer Y`, `Accelerometer Z` (m/s² or g; set `config.ACCEL_UNITS`).
- **Gyroscope:** `Gyroscope X`, `Gyroscope Y`, `Gyroscope Z` (rad/s or deg/s; deg/s is auto-converted).

Shimmer Consensys often uses names like `Low Noise Accelerometer X/Y/Z` and `Gyroscope X/Y/Z` — see `config.SHIMMER_ACCEL_COLS`.

## After you collect data

1. Export your Shimmer (or other IMU) data to CSV and note the exact column names and units.
2. Set `config.DEFAULT_SAMPLE_RATE_HZ` to your logging rate, and adjust `config.LOWPASS_CUTOFF_HZ` and peak-detection thresholds if needed.
3. Run the pipeline on one file; if stroke count looks wrong, tune `ACCEL_PEAK_MIN_HEIGHT` and `ACCEL_PEAK_MIN_DISTANCE`.
4. Use `use_fixed_reach=True` if integrated reach is too noisy; then power uses a fixed arm length.
5. Use `ml_analysis.detect_sei_change_points()` and `ml_analysis.fatigue_clustering()` on `df_features` for change-point and fatigue analysis.
