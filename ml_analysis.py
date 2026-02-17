"""
Machine learning analysis: Change Point Detection (SEI) and Fatigue Clustering.
Use once you have per-stroke features from the pipeline.
"""

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError:
    rpt = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    KMeans = None
    StandardScaler = None


def detect_sei_change_points(
    sei_series: np.ndarray,
    n_bkps: int = 2,
    min_size: int = 2,
) -> list:
    """
    Detect change points in the Stroke Efficiency Index time series.
    Returns list of indices where the mean/variance of SEI changes (stroke indices).
    """
    if rpt is None:
        raise ImportError("Install ruptures: pip install ruptures")
    sei = np.nan_to_num(sei_series, nan=0.0, posinf=0.0, neginf=0.0)
    if len(sei) < min_size * 2:
        return []
    sei_2d = sei.reshape(-1, 1).astype(float)
    algo = rpt.Pelt(model="rbf").fit(sei_2d)
    bkps = algo.predict(pen=1.0)
    # ruptures returns segment end indices; change points are the start of each new segment
    return [int(b) for b in bkps[:-1] if b < len(sei)]


def fatigue_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    features: list = None,
) -> tuple[np.ndarray, object]:
    """
    K-Means clustering on per-stroke features to group strokes into e.g. Fresh / Steady / Fatigued.
    df: DataFrame with columns power_w, spm, reach_m, entry_angle_deg (and optionally sei).
    Returns (cluster_labels, fitted KMeans).
    """
    if KMeans is None or StandardScaler is None:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    features = features or ["power_w", "spm", "reach_m", "entry_angle_deg"]
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].mean())
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km
