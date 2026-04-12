# =============================================================================
# cloud_compute.py -- Heavy ML Computation (Run on Cloud Server)
# =============================================================================
# This script performs ALL computationally expensive operations:
#   - K-Means optimal k search (elbow + silhouette)
#   - K-Means clustering (k=3)
#   - DBSCAN clustering
#   - Hierarchical clustering
#   - Evaluation metrics (silhouette, davies-bouldin)
#   - t-SNE embedding (sampled)
#   - Silhouette per-sample values (sampled)
#
# Results are saved to data/processed/cloud_results.npz so that main.py
# can load them instantly without any heavy computation.
#
# USAGE:
#   python cloud_compute.py
#
# After running, copy the data/processed/ folder back to your local
# machine, then run: python main.py
# =============================================================================

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE

from src.utils import (
    RAW_DATASET_PATH,
    CLUSTERING_FEATURES,
    REFERENCE_COLUMNS,
    DROP_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    print_header,
    print_subheader,
    print_metric,
)

# Output paths
CLOUD_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_results.npz")
CLOUD_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_metrics.json")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def main():
    """Run all heavy computations and save results."""
    start = time.time()

    # =================================================================
    # 1. Load & preprocess
    # =================================================================
    print_header("1. LOADING & PREPROCESSING")

    df = pd.read_csv(RAW_DATASET_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df = df.drop_duplicates().dropna().reset_index(drop=True)
    print(f"  After cleaning: {df.shape[0]:,} rows")

    # Select clustering features
    feature_df = df[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    print(f"  Scaled {X.shape[1]} features with StandardScaler")

    # =================================================================
    # 2. K-Means optimal k search
    # =================================================================
    print_header("2. K-MEANS OPTIMAL K SEARCH")

    k_range = list(range(2, 11))
    inertias = []
    sil_scores = []

    for k in k_range:
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X, labels))
        sil_scores.append(sil)
        elapsed = time.time() - t0
        print(f"  k={k:2d}  |  Inertia: {km.inertia_:>12,.1f}  |  "
              f"Silhouette: {sil:.4f}  |  {elapsed:.1f}s")

    best_k = k_range[np.argmax(sil_scores)]
    print(f"\n  --> Best k = {best_k} (silhouette = {max(sil_scores):.4f})")

    # =================================================================
    # 3. K-Means with best k
    # =================================================================
    print_header(f"3. K-MEANS CLUSTERING (k={best_k})")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    print(f"  Cluster sizes: {dict(zip(*np.unique(kmeans_labels, return_counts=True)))}")

    # =================================================================
    # 4. DBSCAN
    # =================================================================
    print_header("4. DBSCAN CLUSTERING")

    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)

    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int((dbscan_labels == -1).sum())
    print(f"  Clusters: {n_dbscan_clusters}, Noise: {n_noise:,}")

    # =================================================================
    # 5. Hierarchical Clustering
    # =================================================================
    print_header(f"5. HIERARCHICAL CLUSTERING (k={best_k})")

    t0 = time.time()
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_labels = hier.fit_predict(X)
    print(f"  Cluster sizes: {dict(zip(*np.unique(hier_labels, return_counts=True)))}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # =================================================================
    # 6. Evaluation metrics
    # =================================================================
    print_header("6. EVALUATION METRICS")

    metrics = {}

    # K-Means
    km_sil = float(silhouette_score(X, kmeans_labels))
    km_dbi = float(davies_bouldin_score(X, kmeans_labels))
    km_inertia = float(kmeans.inertia_)
    metrics["kmeans"] = {
        "silhouette_score": km_sil,
        "davies_bouldin_index": km_dbi,
        "inertia": km_inertia,
        "n_clusters": best_k,
    }
    print_subheader("K-Means")
    print_metric("Silhouette", km_sil)
    print_metric("Davies-Bouldin", km_dbi)
    print_metric("Inertia", km_inertia, ",.1f")

    # DBSCAN (only if >1 cluster found)
    if n_dbscan_clusters >= 2:
        mask = dbscan_labels != -1
        db_sil = float(silhouette_score(X[mask], dbscan_labels[mask]))
        db_dbi = float(davies_bouldin_score(X[mask], dbscan_labels[mask]))
    else:
        db_sil = -1.0
        db_dbi = float("inf")
    metrics["dbscan"] = {
        "silhouette_score": db_sil,
        "davies_bouldin_index": db_dbi,
        "n_clusters": n_dbscan_clusters,
        "n_noise": n_noise,
    }
    print_subheader("DBSCAN")
    print_metric("Silhouette", db_sil)
    print_metric("Davies-Bouldin", db_dbi)

    # Hierarchical
    h_sil = float(silhouette_score(X, hier_labels))
    h_dbi = float(davies_bouldin_score(X, hier_labels))
    metrics["hierarchical"] = {
        "silhouette_score": h_sil,
        "davies_bouldin_index": h_dbi,
        "n_clusters": best_k,
    }
    print_subheader("Hierarchical")
    print_metric("Silhouette", h_sil)
    print_metric("Davies-Bouldin", h_dbi)

    # Store elbow/silhouette search data
    metrics["k_search"] = {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "best_k": best_k,
    }

    # =================================================================
    # 7. t-SNE embedding (sampled, 5000 points)
    # =================================================================
    print_header("7. t-SNE EMBEDDING")

    tsne_sample_size = 5000
    rng = np.random.RandomState(RANDOM_STATE)
    tsne_indices = rng.choice(len(X), tsne_sample_size, replace=False)
    tsne_indices.sort()

    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=30, n_iter=1000, learning_rate="auto")
    tsne_coords = tsne.fit_transform(X[tsne_indices])
    tsne_labels = kmeans_labels[tsne_indices]
    print(f"  Computed t-SNE on {tsne_sample_size:,} samples in {time.time() - t0:.1f}s")

    # =================================================================
    # 8. Silhouette per-sample values (sampled, 15000 points)
    # =================================================================
    print_header("8. SILHOUETTE DIAGRAM DATA")

    sil_sample_size = 15000
    sil_indices = rng.choice(len(X), sil_sample_size, replace=False)
    sil_indices.sort()

    t0 = time.time()
    sil_values = silhouette_samples(X[sil_indices], kmeans_labels[sil_indices])
    sil_labels = kmeans_labels[sil_indices]
    print(f"  Computed silhouette samples on {sil_sample_size:,} points in {time.time() - t0:.1f}s")

    # =================================================================
    # 9. Cluster profiles
    # =================================================================
    print_header("9. CLUSTER PROFILES (K-Means)")

    profile_df = feature_df.copy()
    profile_df["Cluster"] = kmeans_labels
    cluster_profiles = profile_df.groupby("Cluster")[CLUSTERING_FEATURES].mean()
    print(cluster_profiles.round(3).to_string())

    # =================================================================
    # 10. Save everything
    # =================================================================
    print_header("10. SAVING RESULTS")

    # Save arrays to npz
    np.savez_compressed(
        CLOUD_RESULTS_PATH,
        kmeans_labels=kmeans_labels,
        dbscan_labels=dbscan_labels,
        hier_labels=hier_labels,
        tsne_coords=tsne_coords,
        tsne_labels=tsne_labels,
        sil_values=sil_values,
        sil_labels=sil_labels,
    )
    print(f"  Saved arrays  -> {CLOUD_RESULTS_PATH}")

    # Save metrics to JSON
    with open(CLOUD_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {CLOUD_METRICS_PATH}")

    total = time.time() - start
    print_header("CLOUD COMPUTE COMPLETE")
    print(f"  Total time: {total:.1f}s ({total / 60:.1f} min)")
    print(f"\n  Copy these files to your local machine:")
    print(f"    {CLOUD_RESULTS_PATH}")
    print(f"    {CLOUD_METRICS_PATH}")
    print(f"\n  Then run: python main.py")


if __name__ == "__main__":
    main()
