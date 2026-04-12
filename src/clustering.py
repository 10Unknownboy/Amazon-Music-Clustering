# =============================================================================
# clustering.py — Clustering Algorithms (KMeans, DBSCAN, Hierarchical)
# =============================================================================
# Implements three clustering approaches:
#   1. K-Means with Elbow Method & Silhouette analysis for optimal k
#   2. DBSCAN for density-based, arbitrary-shaped clusters
#   3. Agglomerative Hierarchical Clustering with dendrogram support
# =============================================================================

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from src.utils import (
    KMEANS_K_RANGE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HIERARCHICAL_N_CLUSTERS,
    HIERARCHICAL_LINKAGE,
    RANDOM_STATE,
    print_header,
    print_subheader,
    print_metric,
)


# =============================================================================
# K-Means Clustering
# =============================================================================
def find_optimal_k(X, k_range=None):
    """
    Determine the optimal number of clusters using Elbow Method and
    Silhouette Score.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Scaled feature matrix.
    k_range : range or list of int, optional
        Range of k values to evaluate. Defaults to KMEANS_K_RANGE.

    Returns
    -------
    dict
        Dictionary containing:
        - 'k_range': list of k values tested
        - 'inertias': SSE (inertia) for each k
        - 'silhouette_scores': silhouette score for each k
        - 'optimal_k_elbow': best k from elbow method
        - 'optimal_k_silhouette': best k from silhouette score
        - 'recommended_k': final recommended k
    """
    if k_range is None:
        k_range = KMEANS_K_RANGE

    print_header("FINDING OPTIMAL K FOR K-MEANS")

    inertias = []
    sil_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"  k={k:2d}  |  Inertia: {kmeans.inertia_:>12,.1f}  |  Silhouette: {sil:.4f}")

    # Elbow detection using KneeLocator
    kneedle = KneeLocator(
        list(k_range), inertias, curve="convex", direction="decreasing"
    )
    optimal_k_elbow = kneedle.knee if kneedle.knee else list(k_range)[2]

    # Best k by silhouette score
    best_sil_idx = np.argmax(sil_scores)
    optimal_k_silhouette = list(k_range)[best_sil_idx]

    # Recommendation: prefer elbow, but validate with silhouette
    recommended_k = optimal_k_elbow

    print_subheader("Optimal K Results")
    print_metric("Elbow Method k", optimal_k_elbow)
    print_metric("Best Silhouette k", optimal_k_silhouette)
    print_metric("Recommended k", recommended_k)

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "optimal_k_elbow": optimal_k_elbow,
        "optimal_k_silhouette": optimal_k_silhouette,
        "recommended_k": recommended_k,
    }


def run_kmeans(X, n_clusters):
    """
    Apply K-Means clustering with the specified number of clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple of (np.ndarray, KMeans)
        - Cluster labels for each sample.
        - Fitted KMeans model.
    """
    print_subheader(f"Running K-Means (k={n_clusters})")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)

    print(f"  Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels, kmeans


# =============================================================================
# DBSCAN Clustering
# =============================================================================
def run_dbscan(X, eps=None, min_samples=None):
    """
    Apply DBSCAN clustering for density-based grouping.

    DBSCAN is good for discovering arbitrary-shaped clusters and detecting
    noise/outlier points (labeled as -1).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Scaled feature matrix.
    eps : float, optional
        Maximum distance between two samples for them to be considered
        in the same neighborhood. Defaults to DBSCAN_EPS.
    min_samples : int, optional
        Minimum number of samples in a neighborhood to form a core point.
        Defaults to DBSCAN_MIN_SAMPLES.

    Returns
    -------
    tuple of (np.ndarray, DBSCAN)
        - Cluster labels (-1 = noise).
        - Fitted DBSCAN model.
    """
    if eps is None:
        eps = DBSCAN_EPS
    if min_samples is None:
        min_samples = DBSCAN_MIN_SAMPLES

    print_subheader(f"Running DBSCAN (eps={eps}, min_samples={min_samples})")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise:,}")
    if n_clusters > 0:
        cluster_counts = dict(zip(*np.unique(labels[labels != -1], return_counts=True)))
        print(f"  Cluster sizes  : {cluster_counts}")

    return labels, dbscan


# =============================================================================
# Hierarchical (Agglomerative) Clustering
# =============================================================================
def run_hierarchical(X, n_clusters=None, linkage=None):
    """
    Apply Agglomerative Hierarchical Clustering.

    Creates clusters by iteratively merging the closest pairs of clusters.
    Supports dendrogram visualization for interpreting the merge hierarchy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Scaled feature matrix.
    n_clusters : int, optional
        Number of clusters. Defaults to HIERARCHICAL_N_CLUSTERS.
    linkage : str, optional
        Linkage criterion: 'ward', 'complete', 'average', 'single'.
        Defaults to HIERARCHICAL_LINKAGE.

    Returns
    -------
    tuple of (np.ndarray, AgglomerativeClustering)
        - Cluster labels.
        - Fitted AgglomerativeClustering model.
    """
    if n_clusters is None:
        n_clusters = HIERARCHICAL_N_CLUSTERS
    if linkage is None:
        linkage = HIERARCHICAL_LINKAGE

    print_subheader(f"Running Hierarchical Clustering (n={n_clusters}, linkage={linkage})")

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    print(f"  Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels, model
