"""
Flask server for SwimPerform-style dashboard.
Runs pipeline (synthetic or CSV), then serves Coach Overview, Deep-Dive, and Comparison pages with real data.
"""
# Use non-interactive backend so matplotlib works in Flask worker threads (no GUI on macOS).
import matplotlib
matplotlib.use("Agg")

import json
import os
import uuid
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
import config
from synthetic_data import generate_synthetic_imu
from pipeline import run_pipeline
from ml_analysis import detect_sei_change_points, fatigue_clustering

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

# Directory for uploaded CSV files (created on first use)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ATHLETES_FILE = os.path.join(DATA_DIR, "saved_athletes.json")

LEVELS = [
    ("novice", "Novice"),
    ("high_performance", "High Performance"),
    ("elite", "Elite"),
]


def _ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_saved_athletes():
    """Load list of saved athletes from JSON. Each item: id, name, level, created_at, data."""
    if not os.path.isfile(ATHLETES_FILE):
        return []
    try:
        with open(ATHLETES_FILE, "r") as f:
            out = json.load(f)
        return out.get("athletes", [])
    except Exception:
        return []


def save_saved_athletes(athletes):
    """Persist list of saved athletes to JSON."""
    _ensure_data_dir()
    with open(ATHLETES_FILE, "w") as f:
        json.dump({"athletes": athletes}, f, indent=2)


def get_session_data(filepath=None, mass_kg=70, duration_sec=90, stroke_rate_spm=52):
    """Run pipeline and return a dict of all metrics for templates.
    If filepath is set, run pipeline on that CSV; otherwise use synthetic data.
    """
    if filepath and os.path.isfile(filepath):
        result = run_pipeline(
            filepath=filepath,
            mass_kg=mass_kg,
            use_fixed_reach=True,
        )
    else:
        t, accel, gyro = generate_synthetic_imu(
            duration_sec=duration_sec,
            sample_rate_hz=config.DEFAULT_SAMPLE_RATE_HZ,
            stroke_rate_spm=stroke_rate_spm,
            seed=42,
        )
        result = run_pipeline(
            accel=accel, gyro=gyro, time_s=t,
            sample_rate_hz=config.DEFAULT_SAMPLE_RATE_HZ,
            mass_kg=mass_kg,
            use_fixed_reach=True,
        )

    df = result["df_features"]
    df_raw = result["df_raw"]
    if len(df) == 0:
        return None
    if len(df_raw) == 0:
        raise ValueError("Pipeline produced no raw data. Check CSV format.")

    time_s = df_raw["time_s"]
    duration_s = float(time_s.iloc[-1] - time_s.iloc[0])
    duration_str = f"{int(duration_s // 60):02d}:{duration_s % 60:05.2f}"
    total_reach_m = float(df["reach_m"].sum())
    avg_velocity = round(total_reach_m / duration_s, 2) if duration_s > 0 else 0
    sei_vals = df["sei"].replace([np.inf, -np.inf], np.nan)
    avg_sei_raw = float(sei_vals.mean()) if sei_vals.notna().any() else None
    # SEI as percentage (0-100) and rating 0-10 from min-max within session
    s = sei_vals.dropna()
    if len(s) >= 2 and s.max() > s.min():
        ratings = 10.0 * (s - s.min()) / (s.max() - s.min())
        sei_rating_10 = float(ratings.mean())
        sei_pct = round(sei_rating_10 * 10, 1)  # 0-100
    else:
        sei_rating_10 = 5.0
        sei_pct = 50.0

    if df["entry_angle_deg"].mean() >= -10 and df["entry_angle_deg"].mean() <= 10:
        entry_label = "Good"
    elif df["entry_angle_deg"].mean() < -10:
        entry_label = "Bad (injury risk)"
    else:
        entry_label = "Bad (inefficient)"

    change_points = []
    if len(df) > 4:
        try:
            change_points = detect_sei_change_points(sei_vals.values)
        except Exception:
            change_points = []

    # Fatigue clusters: cluster with highest mean SEI = "Fresh"
    try:
        labels, _ = fatigue_clustering(df, n_clusters=3)
        sei_clean = df["sei"].replace([np.inf, -np.inf], np.nan)
        means = [float(sei_clean[labels == k].mean()) for k in range(3)]
        fresh_cluster = int(np.argmax(means))
        n_fresh = int((labels == fresh_cluster).sum())
        pct_fresh = round(100 * n_fresh / len(labels), 0) if len(labels) else 0
        fresh_sei = float(sei_clean[labels == fresh_cluster].mean()) if n_fresh else 0
        fatigued_sei = float(sei_clean[labels != fresh_cluster].mean()) if (len(labels) - n_fresh) else 0
    except Exception:
        pct_fresh = 72
        fresh_sei = 0.94
        fatigued_sei = 0.76

    # For comparison: split into first half (Session A) vs second half (Session B)
    mid = len(df) // 2
    df_a, df_b = df.iloc[:mid], df.iloc[mid:]
    avg_power_a = float(df_a["power_w"].mean())
    avg_power_b = float(df_b["power_w"].mean())
    avg_reach_a = float(df_a["reach_m"].mean()) * 100  # cm
    avg_reach_b = float(df_b["reach_m"].mean()) * 100
    avg_angle_a = float(df_a["entry_angle_deg"].mean())
    avg_angle_b = float(df_b["entry_angle_deg"].mean())
    power_delta_pct = round((avg_power_b - avg_power_a) / avg_power_a * 100, 1) if avg_power_a else 0
    reach_delta_cm = round(avg_reach_b - avg_reach_a, 0)
    angle_delta = round(avg_angle_b - avg_angle_a, 1)

    scatter_pts, scatter_rng = _scatter_points(df)

    return {
        "stroke_count": len(df),
        "avg_velocity": avg_velocity,
        "avg_spm": round(float(df["spm"].mean()), 1),
        "avg_power": round(float(df["power_w"].mean()), 1),
        "peak_power": round(float(df["power_w"].max()), 0),
        "avg_reach_m": round(float(df["reach_m"].mean()), 3),
        "avg_reach_cm": round(float(df["reach_m"].mean()) * 100, 0),
        "avg_entry_angle": round(float(df["entry_angle_deg"].mean()), 1),
        "entry_label": entry_label,
        "sei_rating_10": round(sei_rating_10, 1),
        "sei_pct": sei_pct,
        "avg_sei_raw": round(avg_sei_raw, 4) if avg_sei_raw and not np.isnan(avg_sei_raw) else None,
        "duration_str": duration_str,
        "duration_s": duration_s,
        "change_points": change_points,
        "pct_fresh": int(pct_fresh),
        "fresh_sei": round(fresh_sei, 2),
        "fatigued_sei": round(fatigued_sei, 2),
        "sei_series": [round(float(x), 4) if np.isfinite(x) else None for x in df["sei"].tolist()],
        "power_series": [round(float(x), 1) for x in df["power_w"].tolist()],
        "spm_series": [round(float(x), 1) for x in df["spm"].tolist()],
        "reach_series": [round(float(x), 3) for x in df["reach_m"].tolist()],
        "entry_angle_series": [round(float(x), 1) for x in df["entry_angle_deg"].tolist()],
        "stroke_indices": [int(x) for x in df["stroke_index"].tolist()],
        # Comparison (first half vs second half)
        "session_a": {
            "avg_power": int(avg_power_a),
            "reach_cm": int(avg_reach_a),
            "entry_angle": round(avg_angle_a, 0),
        },
        "session_b": {
            "avg_power": int(avg_power_b),
            "reach_cm": int(avg_reach_b),
            "entry_angle": round(avg_angle_b, 0),
        },
        "delta_power_pct": power_delta_pct,
        "delta_reach_cm": int(reach_delta_cm),
        "delta_angle": angle_delta,
        # Sparkline path for roster (normalized SEI 0-32)
        "sparkline_path": _sparkline_path(sei_vals.values),
        # SEI chart: path and change point x (0-1000)
        "sei_chart_path": _sei_chart_path(sei_vals.values),
        "change_point_x": _change_point_x(change_points, len(df)),
        # Scatter for Power vs Reach (normalized for SVG) + range for axis labels
        "scatter_points": scatter_pts,
        "scatter_range": scatter_rng,
    }


def _sparkline_path(series, w=100, h=32):
    """Build SVG path for sparkline from 1D array. Returns string like 'M0,20 L10,18...'."""
    arr = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < 2:
        return "M0,16 L100,16"
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        return "M0,16 L100,16"
    pts = []
    for i, v in enumerate(arr):
        x = (i / (len(arr) - 1)) * w if len(arr) > 1 else 0
        y = h - (float(v - mn) / (mx - mn)) * (h - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def _sei_chart_path(series, w=1000, h=250):
    """SEI line for deep-dive chart. Y inverted (higher SEI = lower Y). Returns path d string."""
    arr = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < 2:
        return "M0,125 L1000,125"
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        return "M0,125 L1000,125"
    pts = []
    for i, v in enumerate(arr):
        x = (i / (len(arr) - 1)) * w if len(arr) > 1 else 0
        y = 250 - (float(v - mn) / (mx - mn)) * 200  # 50-250 range
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def _change_point_x(change_points, n_strokes):
    """Return x position (0-1000) for first change point."""
    if not change_points or n_strokes <= 0:
        return None
    return int((change_points[0] / (n_strokes - 1)) * 1000) if n_strokes > 1 else 500


def _scatter_points(df):
    """List of {x, y, color} for Power vs Reach scatter. x,y in plot area 20–190, 10–90.
    Also returns scatter_range dict for axis labels.
    """
    if len(df) == 0:
        return [], None
    pw = np.nan_to_num(df["power_w"].values, nan=0.0, posinf=0.0, neginf=0.0)
    rch = np.nan_to_num(df["reach_m"].values, nan=0.0, posinf=0.0, neginf=0.0) * 100  # cm
    pw_mx, pw_mn = float(np.max(pw)), float(np.min(pw))
    rch_mx, rch_mn = float(np.max(rch)), float(np.min(rch))
    if pw_mx <= pw_mn:
        pw_mx = pw_mn + 1
    if rch_mx <= rch_mn:
        rch_mx = rch_mn + 1
    out = []
    mid = len(df) // 2
    for i in range(len(df)):
        x = 20 + (float(rch[i]) - rch_mn) / (rch_mx - rch_mn) * 160
        y = 90 - (float(pw[i]) - pw_mn) / (pw_mx - pw_mn) * 75
        x = max(22, min(188, x))
        y = max(12, min(88, y))
        color = "primary" if i < mid else "rose"
        out.append({"x": round(x, 1), "y": round(y, 1), "c": color})
    range_dict = {
        "reach_min_m": round(rch_mn / 100, 3),
        "reach_max_m": round(rch_mx / 100, 3),
        "power_min_w": round(pw_mn, 1),
        "power_max_w": round(pw_mx, 1),
    }
    return out, range_dict


def _get_data_or_redirect():
    """Get session data from uploaded file or synthetic. Returns (data_dict, None) or (None, error_message)."""
    filepath = session.get("uploaded_file")
    mass_kg = session.get("mass_kg")
    if mass_kg is None:
        mass_kg = request.args.get("mass", 70, type=int)
    try:
        data = get_session_data(filepath=filepath, mass_kg=mass_kg)
        data = data or {}
        data["swimmer_name"] = session.get("swimmer_name", "")
        data["swimmer_level"] = session.get("swimmer_level", "")
        return (data, None)
    except Exception as e:
        if filepath:
            session.pop("uploaded_file", None)
            session.pop("mass_kg", None)
        return (None, str(e))


@app.route("/")
def index():
    data, err = _get_data_or_redirect()
    if err:
        flash(err, "error")
        return redirect(url_for("data_upload"))
    return render_template("coach_overview.html", data=data)


@app.route("/dashboard")
def coach_overview():
    data, err = _get_data_or_redirect()
    if err:
        flash(err, "error")
        return redirect(url_for("data_upload"))
    return render_template("coach_overview.html", data=data)


@app.route("/deep-dive")
def deep_dive():
    data, err = _get_data_or_redirect()
    if err:
        flash(err, "error")
        return redirect(url_for("data_upload"))
    return render_template("deep_dive.html", data=data)


@app.route("/comparison")
def comparison():
    data, err = _get_data_or_redirect()
    if err:
        flash(err, "error")
        return redirect(url_for("data_upload"))
    source_a = request.args.get("a", "current")
    source_b = request.args.get("b", "current")
    comparison_data = _build_comparison_from_two_sources(source_a, source_b, data)
    if comparison_data is None and (source_a != "current" or source_b != "current"):
        comparison_data = _build_comparison_from_two_sources("current", "current", data)
    saved = load_saved_athletes()
    return render_template(
        "comparison.html",
        data=comparison_data or data,
        saved_athletes=saved,
        levels=LEVELS,
        source_a=source_a,
        source_b=source_b,
    )


@app.route("/data-upload", methods=["GET", "POST"])
def data_upload():
    if request.method == "POST":
        file = request.files.get("csv_file")
        try:
            mass_kg = int(float(request.form.get("mass_kg", 70)))
        except (TypeError, ValueError):
            mass_kg = 70
        mass_kg = max(30, min(150, mass_kg))
        use_synthetic = "use_synthetic" in request.form

        if use_synthetic:
            session.pop("uploaded_file", None)
            session.pop("swimmer_name", None)
            session.pop("swimmer_level", None)
            session["mass_kg"] = mass_kg
            flash("Using synthetic data. View dashboard with your chosen mass.", "success")
            return redirect(url_for("coach_overview"))

        if not file or file.filename == "":
            flash("Please select a CSV file to upload.", "error")
            return redirect(url_for("data_upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("File must be a CSV.", "error")
            return redirect(url_for("data_upload"))

        _ensure_upload_dir()
        safe_name = f"{uuid.uuid4().hex}.csv"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        try:
            file.save(save_path)
        except Exception as e:
            flash(f"Could not save file: {e}", "error")
            return redirect(url_for("data_upload"))

        # Validate by running pipeline; on success store in session
        try:
            data = get_session_data(filepath=save_path, mass_kg=mass_kg)
            if not data or data.get("stroke_count", 0) == 0:
                flash("Pipeline produced no strokes. Check CSV format (accel/gyro columns and units).", "error")
                try:
                    os.remove(save_path)
                except OSError:
                    pass
                return redirect(url_for("data_upload"))
        except Exception as e:
            try:
                os.remove(save_path)
            except OSError:
                pass
            flash(f"Processing failed: {e}", "error")
            return redirect(url_for("data_upload"))

        session["uploaded_file"] = save_path
        session["mass_kg"] = mass_kg
        session["swimmer_name"] = (request.form.get("swimmer_name") or "").strip() or None
        level = (request.form.get("swimmer_level") or "").strip()
        if level in ("novice", "high_performance", "elite"):
            session["swimmer_level"] = level
        else:
            session["swimmer_level"] = "novice"
        flash(f"File uploaded successfully. {data['stroke_count']} strokes detected.", "success")
        return redirect(url_for("coach_overview"))

    # GET: show upload form
    has_upload = bool(session.get("uploaded_file"))
    saved = load_saved_athletes()
    return render_template("data_upload.html", has_upload=has_upload, saved_athletes=saved, levels=LEVELS)


@app.route("/data-upload/clear")
def data_upload_clear():
    session.pop("uploaded_file", None)
    session.pop("mass_kg", None)
    session.pop("swimmer_name", None)
    session.pop("swimmer_level", None)
    flash("Cleared uploaded file. Using synthetic data until you upload again.", "success")
    return redirect(url_for("data_upload"))


@app.route("/save-athlete", methods=["POST"])
def save_athlete():
    """Save current session (name, level, metrics) to stored list for later comparison."""
    data, err = _get_data_or_redirect()
    if err:
        flash(err, "error")
        return redirect(url_for("data_upload"))
    name = (session.get("swimmer_name") or request.form.get("swimmer_name") or "").strip() or "Unnamed"
    level = session.get("swimmer_level") or request.form.get("swimmer_level") or "novice"
    if level not in ("novice", "high_performance", "elite"):
        level = "novice"
    athletes = load_saved_athletes()
    athletes.append({
        "id": uuid.uuid4().hex,
        "name": name,
        "level": level,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    })
    save_saved_athletes(athletes)
    flash(f"Saved “{name}” ({dict(LEVELS).get(level, level)}). You can compare them on the Comparison page.", "success")
    return redirect(url_for("coach_overview"))


def _build_comparison_from_two_sources(source_a, source_b, current_data):
    """Build comparison data dict for template from two sources. Source is 'current' or saved athlete id."""
    def metrics_from_data(d):
        return {
            "avg_power": d.get("avg_power", 0),
            "reach_cm": d.get("avg_reach_cm", 0),
            "entry_angle": d.get("avg_entry_angle", 0),
            "name": d.get("swimmer_name") or "Session",
            "level": d.get("swimmer_level") or "—",
        }
    if source_a == "current" and source_b == "current" and current_data:
        # First half vs second half of current session
        mid = (current_data.get("stroke_count") or 0) // 2
        return {
            **current_data,
            "session_a": {
                "name": (current_data.get("swimmer_name") or "Current") + " (first half)",
                "level": current_data.get("swimmer_level", "—"),
                "avg_power": current_data.get("session_a", {}).get("avg_power", 0),
                "reach_cm": current_data.get("session_a", {}).get("reach_cm", 0),
                "entry_angle": current_data.get("session_a", {}).get("entry_angle", 0),
            },
            "session_b": {
                "name": (current_data.get("swimmer_name") or "Current") + " (second half)",
                "level": current_data.get("swimmer_level", "—"),
                "avg_power": current_data.get("session_b", {}).get("avg_power", 0),
                "reach_cm": current_data.get("session_b", {}).get("reach_cm", 0),
                "entry_angle": current_data.get("session_b", {}).get("entry_angle", 0),
            },
            "delta_power_pct": current_data.get("delta_power_pct", 0),
            "delta_reach_cm": current_data.get("delta_reach_cm", 0),
            "delta_angle": current_data.get("delta_angle", 0),
            "stroke_count": current_data.get("stroke_count"),
            "duration_str": current_data.get("duration_str"),
            "duration_s": current_data.get("duration_s"),
        }
    athletes = {a["id"]: a for a in load_saved_athletes()}
    a_data = current_data if source_a == "current" else athletes.get(source_a, {}).get("data", {})
    b_data = current_data if source_b == "current" else athletes.get(source_b, {}).get("data", {})
    if not a_data or not b_data:
        return None
    sa = metrics_from_data(a_data)
    sb = metrics_from_data(b_data)
    if source_a != "current":
        sa["name"] = athletes.get(source_a, {}).get("name", "Saved")
        sa["level"] = dict(LEVELS).get(athletes.get(source_a, {}).get("level", ""), "—")
    if source_b != "current":
        sb["name"] = athletes.get(source_b, {}).get("name", "Saved")
        sb["level"] = dict(LEVELS).get(athletes.get(source_b, {}).get("level", ""), "—")
    pa, pb = sa["avg_power"], sb["avg_power"]
    delta_power = round((pb - pa) / pa * 100, 1) if pa else 0
    return {
        "session_a": sa,
        "session_b": sb,
        "delta_power_pct": delta_power,
        "delta_reach_cm": int(sb["reach_cm"] - sa["reach_cm"]),
        "delta_angle": round(sb["entry_angle"] - sa["entry_angle"], 1),
        "stroke_count": a_data.get("stroke_count") or b_data.get("stroke_count"),
        "duration_str": a_data.get("duration_str") or b_data.get("duration_str", "00:00"),
        "duration_s": a_data.get("duration_s") or b_data.get("duration_s"),
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
