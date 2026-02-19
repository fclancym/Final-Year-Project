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

# Page config
st.set_page_config(
    page_title="Swimming IMU Analytics",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for black/red/white theme
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0a0a; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111111; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; }
    p, label, span { color: #e0e0e0 !important; }
    
    /* Metric cards */
    [data-testid="stMetricValue"] { color: #ff3333 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    
    /* Input focus */
    .stSlider > div > div > div { background: #ff3333 !important; }
    
    /* Expander */
    .streamlit-expanderHeader { background-color: #1a1a1a !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)


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


def plot_stroke_rate_power(df):
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
    fig.update_layout(
        title="Stroke Rate & Power",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,20,0.8)",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Stroke", gridcolor="#333333"),
        yaxis=dict(title="Power (W)", gridcolor="#333333"),
        yaxis2=dict(title="Stroke Rate (st/min)", gridcolor="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
    )
    fig.update_yaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    fig.update_xaxes(title_font=dict(color="#cccccc"), tickfont=dict(color="#aaaaaa"))
    return fig


def plot_entry_angle(df):
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
    return fig


def plot_sei(df):
    """Stroke Efficiency Index over time."""
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
    return fig


def main():
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

    # KPI row
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

    # Main chart: Stroke Rate & Power
    st.plotly_chart(plot_stroke_rate_power(df), use_container_width=True)

    # Second row: Entry angle + SEI
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_entry_angle(df), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_sei(df), use_container_width=True)

    # Expandable: raw data table
    with st.expander("View per-stroke data"):
        st.dataframe(df, use_container_width=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("FYP: ML Analytics for Stroke Count, Technique and Fatigue Detection")


if __name__ == "__main__":
    main()
