"""
FYP Swimming IMU Dashboard - Streamlit web app.
Black, red, white theme. Uses synthetic data by default; upload CSV for real data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from synthetic_data import generate_synthetic_imu
from pipeline import run_pipeline
from ml_analysis import detect_sei_change_points, fatigue_clustering

# Page config
st.set_page_config(
    page_title="Swimming IMU Analytics",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark mode toggle (persists in session)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def get_theme_css(dark_mode):
    """Return CSS for light or dark theme."""
    if dark_mode:
        return """
        <style>
            .stApp { background-color: #0a0a0a; }
            [data-testid="stSidebar"] { background-color: #111111; }
            h1, h2, h3 { color: #ffffff !important; }
            p, label, span { color: #e0e0e0 !important; }
            [data-testid="stMetricValue"] { color: #ff3333 !important; font-weight: bold; }
            [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
            .streamlit-expanderHeader { background-color: #1a1a1a !important; color: #fff !important; }
        </style>
        """
    return """
    <style>
        [data-testid="stMetricValue"] { color: #cc0000 !important; font-weight: bold; }
    </style>
    """


def render_metric_info(ml_goal, technique, libraries, input_features, output_result):
    """Render a simplified info table for a metric/ML technique."""
    st.markdown("#### Methodology")
    st.markdown(f"**ML Goal:** {ml_goal}")
    st.markdown(f"**Technique:** {technique}")
    st.markdown(f"**Libraries:** {libraries}")
    st.markdown(f"**Input Features (per stroke):** {input_features}")
    st.markdown(f"**Output Result:** {output_result}")


def load_data():
    """Load synthetic or uploaded CSV. Returns (accel, gyro, time_s, sample_rate_hz) or filepath."""
    data_source = st.sidebar.radio(
        "Data source",
        ["Synthetic (demo)", "Upload CSV"],
        help="Use synthetic data for testing, or upload your Shimmer IMU export.",
    )
    if data_source == "Synthetic (demo)":
        duration = st.sidebar.slider("Duration (seconds)", 60, 180, 90)
        stroke_rate = st.sidebar.slider("Stroke rate (SPM)", 40, 70, 52)
        t, accel, gyro = generate_synthetic_imu(
            duration_sec=duration,
            sample_rate_hz=config.DEFAULT_SAMPLE_RATE_HZ,
            stroke_rate_spm=stroke_rate,
            seed=42,
        )
        return {"accel": accel, "gyro": gyro, "time_s": t, "sample_rate_hz": config.DEFAULT_SAMPLE_RATE_HZ}
    else:
        uploaded = st.sidebar.file_uploader("Upload IMU CSV", type=["csv"])
        if uploaded is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                return {"filepath": tmp.name}
        return None


def get_weight_input():
    """Weight input in sidebar. Returns mass_kg."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Swimmer")
    weight_kg = st.sidebar.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=150,
        value=70,
        step=1,
        help="Used for power calculation: P = (m × a) × reach × stroke rate",
    )
    return weight_kg


def run_analysis(data, mass_kg):
    """Run pipeline and return result."""
    if "filepath" in data:
        with st.spinner("Processing uploaded file..."):
            result = run_pipeline(filepath=data["filepath"], mass_kg=mass_kg, use_fixed_reach=True)
    else:
        with st.spinner("Processing data..."):
            result = run_pipeline(
                accel=data["accel"],
                gyro=data["gyro"],
                time_s=data["time_s"],
                sample_rate_hz=data["sample_rate_hz"],
                mass_kg=mass_kg,
                use_fixed_reach=True,
            )
    return result


def plot_stroke_rate(df, dark_mode=False):
    """Stroke rate (SPM) over strokes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["stroke_index"],
            y=df["spm"],
            mode="lines+markers",
            line=dict(color="#00cc66", width=2),
            marker=dict(size=5),
            name="Stroke Rate (st/min)",
        )
    )
    if dark_mode:
        fig.update_layout(
            title="Strokes Per Minute",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="SPM (st/min)", gridcolor="#333333"),
            showlegend=False,
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        fig.update_layout(
            title="Strokes Per Minute",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="SPM (st/min)", gridcolor="#dddddd"),
            showlegend=False,
        )
    return fig


def plot_power(df, dark_mode=False):
    """Power per stroke."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["stroke_index"],
            y=df["power_w"],
            name="Power (W)",
            marker_color="#cc0000",
            opacity=0.7,
        )
    )
    if dark_mode:
        fig.update_layout(
            title="Power per Stroke",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="Power (W)", gridcolor="#333333"),
            showlegend=False,
            bargap=0.2,
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        fig.update_layout(
            title="Power per Stroke",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="Power (W)", gridcolor="#dddddd"),
            showlegend=False,
            bargap=0.2,
        )
    return fig


def plot_stroke_rate_power(df, dark_mode=False):
    """EO-style stacked bar + stroke rate line chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Power bars (single hand for now)
    fig.add_trace(
        go.Bar(
            x=df["stroke_index"],
            y=df["power_w"],
            name="Power (W)",
            marker_color="#ff3333",
            opacity=0.7,
        ),
        secondary_y=False,
    )
    # Stroke rate line
    fig.add_trace(
        go.Scatter(
            x=df["stroke_index"],
            y=df["spm"],
            name="Stroke Rate (st/min)",
            mode="lines+markers",
            line=dict(color="#00cc66", width=2),
            marker=dict(size=4),
        ),
        secondary_y=True,
    )
    layout_kw = dict(
        title="Stroke Rate & Power",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
    )
    if dark_mode:
        layout_kw.update(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="Power (W)", gridcolor="#333333"),
            yaxis2=dict(title="Stroke Rate (st/min)", gridcolor="#333333"),
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        layout_kw.update(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="Power (W)", gridcolor="#dddddd"),
            yaxis2=dict(title="Stroke Rate (st/min)", gridcolor="#dddddd"),
        )
    fig.update_layout(**layout_kw)
    return fig


def plot_entry_angle(df, dark_mode=False):
    """Hand entry angle over strokes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["stroke_index"],
            y=df["entry_angle_deg"],
            mode="lines+markers",
            line=dict(color="#ff6666", width=2),
            marker=dict(size=5),
            name="Entry angle (°)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", opacity=0.5)
    if dark_mode:
        fig.update_layout(
            title="Hand Entry Angle",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="Angle (°)", gridcolor="#333333"),
            showlegend=False,
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        fig.update_layout(
            title="Hand Entry Angle",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="Angle (°)", gridcolor="#dddddd"),
            showlegend=False,
        )
    return fig


def plot_reach(df, dark_mode=False):
    """Stroke reach over strokes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["stroke_index"],
            y=df["reach_m"],
            mode="lines+markers",
            line=dict(color="#3366cc", width=2),
            marker=dict(size=4),
            name="Reach (m)",
        )
    )
    if dark_mode:
        fig.update_layout(
            title="Stroke Reach",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="Reach (m)", gridcolor="#333333"),
            showlegend=False,
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        fig.update_layout(
            title="Stroke Reach",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="Reach (m)", gridcolor="#dddddd"),
            showlegend=False,
        )
    return fig


def plot_sei(df, dark_mode=False, change_points=None):
    """Stroke Efficiency Index over time, optionally with change points."""
    sei = df["sei"].replace([np.inf, -np.inf], np.nan)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["stroke_index"],
            y=sei,
            mode="lines+markers",
            line=dict(color="#ffaa00", width=2),
            marker=dict(size=4),
            name="SEI",
        )
    )
    if change_points:
        for cp in change_points:
            if 0 <= cp < len(df):
                fig.add_vline(x=cp, line_dash="dash", line_color="#ff3333", opacity=0.7)
    if dark_mode:
        fig.update_layout(
            title="Stroke Efficiency Index (Reach / Power)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Stroke", gridcolor="#333333"),
            yaxis=dict(title="SEI", gridcolor="#333333"),
            showlegend=False,
        )
        fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
        fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    else:
        fig.update_layout(
            title="Stroke Efficiency Index (Reach / Power)",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(250,250,250,0.9)",
            font=dict(color="#1a1a1a"),
            xaxis=dict(title="Stroke", gridcolor="#dddddd"),
            yaxis=dict(title="SEI", gridcolor="#dddddd"),
            showlegend=False,
        )
    return fig


def main():
    # Dark mode toggle - top left corner
    top_left, _ = st.columns([1, 8])
    with top_left:
        dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode, key="dark_toggle")
        st.session_state.dark_mode = dark_mode

    # Apply theme CSS
    st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)

    st.title("🏊 Swimming IMU Analytics")
    st.markdown("Stroke count, power, reach, entry angle & efficiency — from your IMU data.")

    # Sidebar: data + weight
    st.sidebar.header("Data")
    data = load_data()
    weight_kg = get_weight_input()

    if data is None:
        st.info("Upload a CSV file to analyse your swim data.")
        return

    # Run pipeline
    result = run_analysis(data, weight_kg)
    df = result["df_features"]
    df_raw = result["df_raw"]

    if len(df) == 0:
        st.warning("No strokes detected. Check your data or adjust peak detection thresholds.")
        return

    # Session summary
    duration_s = df_raw["time_s"].iloc[-1] - df_raw["time_s"].iloc[0]
    duration_str = f"{int(duration_s // 60):02d}:{int(duration_s % 60):02d}"

    # KPI row (always visible)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Strokes", len(df))
    with col2:
        st.metric("Avg Power (W)", f"{df['power_w'].mean():.1f}")
    with col3:
        st.metric("Avg Stroke Rate", f"{df['spm'].mean():.1f} st/min")
    with col4:
        st.metric("Avg Reach (m)", f"{df['reach_m'].mean():.2f}")
    with col5:
        sei_mean = df["sei"].replace([np.inf, -np.inf], np.nan).mean()
        st.metric("Avg SEI", f"{sei_mean:.4f}" if not np.isnan(sei_mean) else "—")
    with col6:
        st.metric("Time", duration_str)

    st.markdown("---")

    # Tabs for each metric
    tab_overview, tab_spm, tab_power, tab_reach, tab_angle, tab_sei, tab_fatigue = st.tabs([
        "Overview",
        "Stroke Rate (SPM)",
        "Power",
        "Stroke Reach",
        "Hand Entry Angle",
        "Stroke Efficiency (SEI)",
        "Fatigue Detection",
    ])

    with tab_overview:
        st.plotly_chart(plot_stroke_rate_power(df, dark_mode), use_container_width=True)
        with st.expander("View per-stroke data"):
            st.dataframe(df, use_container_width=True)

    with tab_spm:
        st.plotly_chart(plot_stroke_rate(df, dark_mode), use_container_width=True)
        st.markdown("---")
        render_metric_info(
            ml_goal="Track pacing and consistency; early marker of fatigue (SPM may increase as efficiency drops).",
            technique="Peak detection on Z-axis acceleration",
            libraries="scipy.signal.find_peaks",
            input_features="Filtered accelerometer Z-axis",
            output_result="Stroke count and stroke rate (SPM) per stroke.",
        )

    with tab_power:
        st.plotly_chart(plot_power(df, dark_mode), use_container_width=True)
        st.markdown("---")
        render_metric_info(
            ml_goal="Measure mechanical output per stroke.",
            technique="Physics model: P = (m × a) × reach × stroke rate",
            libraries="NumPy, pandas",
            input_features="Mass (kg), peak acceleration (m/s²), reach (m), stroke rate (1/s)",
            output_result="Power in Watts per stroke.",
        )

    with tab_reach:
        st.plotly_chart(plot_reach(df, dark_mode), use_container_width=True)
        st.markdown("---")
        render_metric_info(
            ml_goal="Measure effective distance per stroke; direct indicator of efficiency.",
            technique="Double integration of acceleration with ZUPT (or fixed arm length fallback)",
            libraries="NumPy, scipy",
            input_features="Z-axis acceleration (fused with gyro orientation)",
            output_result="Stroke reach in metres per stroke.",
        )

    with tab_angle:
        st.plotly_chart(plot_entry_angle(df, dark_mode), use_container_width=True)
        st.markdown("---")
        render_metric_info(
            ml_goal="Classify entry as Good (flat), Bad (injury risk), or Inefficient.",
            technique="Pitch/Roll from sensor fusion at water entry; future: ML classifier",
            libraries="ahrs / custom Madgwick, scikit-learn (planned)",
            input_features="Pitch and roll (degrees) at stroke entry timestamp",
            output_result="Entry angle per stroke; future: Good / Bad / Inefficient labels.",
        )

    with tab_sei:
        sei_clean = df["sei"].replace([np.inf, -np.inf], np.nan)
        change_points = detect_sei_change_points(sei_clean.values) if len(df) > 4 else []
        st.plotly_chart(plot_sei(df, dark_mode, change_points), use_container_width=True)
        if change_points:
            st.info(f"**Change points detected** at strokes: {change_points}. Efficiency may have shifted at these points.")
        st.markdown("---")
        render_metric_info(
            ml_goal="Pinpoint where efficiency statistically drops during the swim.",
            technique="Change Point Detection",
            libraries="ruptures",
            input_features="Time-series of Stroke Efficiency Index (SEI = Reach / Power)",
            output_result="Stroke numbers where SEI mean/variance shifts; targeted feedback (e.g. 'Performance declined after stroke 75').",
        )

    with tab_fatigue:
        try:
            labels, _ = fatigue_clustering(df, n_clusters=3)
            df_fatigue = df.copy()
            df_fatigue["Cluster"] = labels
            cluster_names = {0: "Fresh", 1: "Steady", 2: "Fatigued"}
            df_fatigue["Phase"] = df_fatigue["Cluster"].map(cluster_names)
            fig_fatigue = px.scatter(
                df_fatigue, x="stroke_index", y="power_w",
                color="Phase", color_discrete_map={"Fresh": "#00cc66", "Steady": "#ffaa00", "Fatigued": "#cc0000"},
            )
            fig_fatigue.update_layout(
                title="Fatigue Clustering (Power vs Stroke)",
                xaxis_title="Stroke",
                yaxis_title="Power (W)",
            )
            if dark_mode:
                fig_fatigue.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,20,0.8)",
                    font=dict(color="#ffffff"),
                )
            st.plotly_chart(fig_fatigue, use_container_width=True)
            st.caption("Strokes grouped into Fresh / Steady / Fatigued based on Power, SPM, Reach, and Entry Angle.")
        except Exception as e:
            st.warning(f"Clustering could not run: {e}")
        st.markdown("---")
        render_metric_info(
            ml_goal="Detect fatigue signatures without manual labelling.",
            technique="K-Means Clustering",
            libraries="sklearn.cluster",
            input_features="Normalized: Power (P), Stroke Rate (SR), Reach (D), Entry Angle (A)",
            output_result="Strokes grouped into clusters (e.g. Fresh, Steady-State, Fatigued) based on performance characteristics.",
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("FYP: ML Analytics for Stroke Count, Technique and Fatigue Detection")


if __name__ == "__main__":
    main()
