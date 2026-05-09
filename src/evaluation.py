# =============================================================================
# evaluation.py — Cluster Evaluation & Interpretation
# =============================================================================
# Provides metrics to evaluate clustering quality and functions to interpret
# what each cluster represents by profiling feature means.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.utils import print_header, print_subheader, print_metric


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_silhouette(X, labels):
    """
    Compute the Silhouette Score for the clustering result.

    The silhouette score measures how similar each sample is to its own
    cluster compared to other clusters. Range: [-1, 1]. Higher is better.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels.

    Returns
    -------
    float
        Silhouette score.
    """
    # Need at least 2 unique labels (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if len(unique_labels) < 2:
        print("[WARNING] Cannot compute silhouette with fewer than 2 clusters.")
        return -1.0

    return silhouette_score(X, labels)


def compute_davies_bouldin(X, labels):
    """
    Compute the Davies-Bouldin Index.

    Measures the average similarity ratio of each cluster with the cluster
    most similar to it. Lower is better (better separation).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels.

    Returns
    -------
    float
        Davies-Bouldin index.
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if len(unique_labels) < 2:
        print("[WARNING] Cannot compute Davies-Bouldin with fewer than 2 clusters.")
        return float("inf")

    # Filter out noise points for DBI calculation
    mask = labels != -1
    return davies_bouldin_score(X[mask], labels[mask])


def compute_inertia(model):
    """
    Extract the inertia (SSE) from a fitted KMeans model.

    Parameters
    ----------
    model : KMeans
        Fitted KMeans model.

    Returns
    -------
    float
        Inertia value, or None if model has no inertia attribute.
    """
    if hasattr(model, "inertia_"):
        return model.inertia_
    return None


def evaluate_all(X, labels, model=None):
    """
    Run all evaluation metrics and print a summary report.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    model : sklearn estimator, optional
        Fitted clustering model (used to extract inertia for KMeans).

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    print_subheader("Cluster Evaluation Metrics")

    metrics = {}

    # Silhouette Score
    sil = compute_silhouette(X, labels)
    metrics["silhouette_score"] = sil
    print_metric("Silhouette Score", sil)

    # Davies-Bouldin Index
    dbi = compute_davies_bouldin(X, labels)
    metrics["davies_bouldin_index"] = dbi
    print_metric("Davies-Bouldin Index", dbi)

    # Inertia (KMeans only)
    if model is not None:
        inertia = compute_inertia(model)
        if inertia is not None:
            metrics["inertia"] = inertia
            print_metric("Inertia (SSE)", inertia, fmt=",.1f")

    # Cluster count and sizes
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((labels == -1).sum())
    metrics["n_clusters"] = n_clusters
    metrics["n_noise_points"] = n_noise
    print_metric("Number of Clusters", n_clusters)
    if n_noise > 0:
        print_metric("Noise Points", f"{n_noise:,}")

    return metrics


# ---------------------------------------------------------------------------
# Cluster profiling & interpretation
# ---------------------------------------------------------------------------
def profile_clusters(df, labels, features=None):
    """
    Compute mean feature values per cluster to understand what each
    cluster represents musically.

    Parameters
    ----------
    df : pd.DataFrame
        Original (unscaled) feature DataFrame.
    labels : array-like of shape (n_samples,)
        Cluster labels.
    features : list of str, optional
        Columns to include in profiling. Defaults to all columns in df.

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster labels as rows and features as columns,
        showing the mean value of each feature per cluster.
    """
    if features is None:
        features = list(df.columns)

    profile_df = df[features].copy()
    profile_df["Cluster"] = labels

    # Exclude noise points from profiling
    profile_df = profile_df[profile_df["Cluster"] != -1]

    cluster_means = profile_df.groupby("Cluster").mean()

    print_header("CLUSTER PROFILES")
    print(cluster_means.round(3).to_string())

    return cluster_means


def name_cluster(row):
    """
    Generate a descriptive cluster name based on the top 2 dominant
    audio features of that cluster.

    Uses threshold-based rules on the mean feature values to identify
    the most distinguishing audio traits of each cluster.

    Parameters
    ----------
    row : pd.Series
        Mean feature values for a single cluster.

    Returns
    -------
    str
        A descriptive label like 'high-energy + danceable'.
    """
    traits = []
    if row.get('speechiness', 0) > 0.5:
        traits.append('speech-heavy')
    if row.get('acousticness', 0) > 0.6:
        traits.append('acoustic')
    if row.get('energy', 0) > 0.65:
        traits.append('high-energy')
    if row.get('energy', 1) < 0.35:
        traits.append('low-energy')
    if row.get('danceability', 0) > 0.7:
        traits.append('danceable')
    if row.get('instrumentalness', 0) > 0.3:
        traits.append('instrumental')
    if row.get('valence', 0) > 0.7:
        traits.append('upbeat')
    if row.get('valence', 1) < 0.3:
        traits.append('melancholic')
    if row.get('tempo', 0) > 130:
        traits.append('fast-tempo')
    if row.get('liveness', 0) > 0.4:
        traits.append('live-feel')
    return ' + '.join(traits[:2]) if traits else 'balanced'


def interpret_clusters(cluster_profiles):
    """
    Generate human-readable labels for each cluster using the new
    threshold-based naming system.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Mean feature values per cluster (from profile_clusters).

    Returns
    -------
    dict
        Mapping of cluster_id → descriptive label string.
    """
    print_subheader("Cluster Interpretations")

    interpretations = {}

    for cluster_id in cluster_profiles.index:
        row = cluster_profiles.loc[cluster_id]
        label = name_cluster(row)
        interpretations[cluster_id] = label
        print(f"  Cluster {cluster_id}: {label}")

    return interpretations














