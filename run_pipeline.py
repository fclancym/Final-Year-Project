"""
Run the full pipeline on synthetic data (or pass a CSV path).
Usage:
  python run_pipeline.py
  python run_pipeline.py --csv path/to/your_imu_data.csv
  python run_pipeline.py --csv data.csv --mass 75 --fixed-reach
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="FYP Swimming IMU analysis pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to IMU CSV. If omitted, use synthetic data.")
    parser.add_argument("--mass", type=float, default=None, help="Swimmer mass (kg) for power calculation")
    parser.add_argument("--fixed-reach", action="store_true", help="Use fixed arm length instead of integrated reach")
    parser.add_argument("--no-plot", action="store_true", help="Skip showing plots (save only)")
    parser.add_argument("--out", type=str, default=None, help="Save features to this CSV path")
    args = parser.parse_args()

    from pipeline import run_pipeline

    if args.csv:
        result = run_pipeline(filepath=args.csv, mass_kg=args.mass, use_fixed_reach=args.fixed_reach)
    else:
        from synthetic_data import generate_synthetic_imu
        import config
        print("No CSV provided. Generating synthetic IMU data (60 s, 50 SPM)...")
        t, accel, gyro = generate_synthetic_imu(
            duration_sec=60.0,
            sample_rate_hz=config.DEFAULT_SAMPLE_RATE_HZ,
            stroke_rate_spm=50.0,
            seed=42,
        )
        result = run_pipeline(
            accel=accel,
            gyro=gyro,
            time_s=t,
            sample_rate_hz=config.DEFAULT_SAMPLE_RATE_HZ,
            mass_kg=args.mass,
            use_fixed_reach=args.fixed_reach,
        )

    df = result["df_features"]
    print(f"Detected {len(df)} strokes. Sample rate: {result['sample_rate_hz']:.1f} Hz")
    if len(df) > 0:
        print(df[["stroke_index", "time_s", "spm", "reach_m", "power_w", "entry_angle_deg", "sei"]].head(10).to_string())

    if args.out and len(df) > 0:
        df.to_csv(args.out, index=False)
        print(f"Saved features to {args.out}")

    if not args.no_plot and result["figures"]:
        import matplotlib.pyplot as plt
        for i, fig in enumerate(result["figures"]):
            fig.savefig(f"pipeline_fig_{i+1}.png", dpi=120)
            print(f"Saved pipeline_fig_{i+1}.png")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
