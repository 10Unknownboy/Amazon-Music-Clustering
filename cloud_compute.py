# =============================================================================
# cloud_compute.py -- Heavy ML Computation (GPU-Accelerated for P100)
# =============================================================================
# Run this on a cloud server with NVIDIA P100 GPU and RAPIDS cuML installed.
# Falls back to scikit-learn CPU if cuML is not available.
#
# GPU Setup (Colab / cloud):
#   !pip install cuml-cu11 cudf-cu11    # For CUDA 11.x
#   OR
#   !pip install cuml-cu12 cudf-cu12    # For CUDA 12.x
#
# USAGE:
#   python cloud_compute.py
#
# OUTPUT:
#   data/processed/cloud_results.npz   -- arrays (labels, embeddings)
#   data/processed/cloud_metrics.json  -- evaluation metrics
#
# Copy outputs to your local machine, then run: python main.py
# =============================================================================

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    RAW_DATASET_PATH,
    CLUSTERING_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    print_header,
    print_subheader,
    print_metric,
)

# ---------------------------------------------------------------------------
# GPU Detection & Import Strategy
# ---------------------------------------------------------------------------
USE_GPU = False

try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.cluster import AgglomerativeClustering as cuHierarchical
    from cuml.manifold import TSNE as cuTSNE
    from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import cudf
    USE_GPU = True
    print("[GPU] RAPIDS cuML detected -- using GPU acceleration (P100)")
except ImportError:
    print("[CPU] RAPIDS cuML not found -- falling back to scikit-learn (CPU)")

# CPU fallbacks (always imported for metrics not in cuML)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples
from joblib import Parallel, delayed

# Output paths
CLOUD_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_results.npz")
CLOUD_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_metrics.json")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
class Timer:
    """Context manager for timestamped phase tracking."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] START  {self.label}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] DONE   {self.label} ({self.elapsed:.2f}s)")


def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# GPU-aware helpers
# ---------------------------------------------------------------------------
def fit_kmeans(X, k):
    """Fit KMeans using GPU if available, else CPU."""
    if USE_GPU:
        model = cuKMeans(n_clusters=k, random_state=RANDOM_STATE,
                         max_iter=300, n_init=10)
        labels = model.fit_predict(X)
        return np.asarray(labels), float(model.inertia_)
    else:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                       n_init=10, max_iter=300)
        labels = model.fit_predict(X)
        return labels, float(model.inertia_)


def compute_silhouette(X, labels):
    """Compute silhouette score using GPU if available."""
    if USE_GPU:
        return float(cu_silhouette_score(X, labels))
    else:
        return float(silhouette_score(X, labels))


def fit_dbscan(X, eps=1.5, min_samples=10):
    """Fit DBSCAN using GPU if available."""
    if USE_GPU:
        model = cuDBSCAN(eps=eps, min_samples=min_samples)
        labels = np.asarray(model.fit_predict(X))
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    return labels


def fit_hierarchical(X, n_clusters):
    """Fit Agglomerative Clustering using GPU if available."""
    if USE_GPU:
        # cuML AgglomerativeClustering supports n_clusters and connectivity
        model = cuHierarchical(n_clusters=n_clusters)
        labels = np.asarray(model.fit_predict(X))
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X)
    return labels


def compute_tsne(X, n_components=2, perplexity=30, n_iter=1000):
    """Compute t-SNE using GPU if available."""
    if USE_GPU:
        tsne = cuTSNE(n_components=n_components, random_state=RANDOM_STATE,
                      perplexity=perplexity, n_iter=n_iter,
                      learning_rate="auto")
        coords = np.asarray(tsne.fit_transform(X))
    else:
        tsne = TSNE(n_components=n_components, random_state=RANDOM_STATE,
                    perplexity=perplexity, n_iter=n_iter,
                    learning_rate="auto")
        coords = tsne.fit_transform(X)
    return coords


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run all heavy computations with GPU acceleration."""
    pipeline_start = time.time()
    print("=" * 72)
    print(f"  CLOUD COMPUTE -- Amazon Music Clustering")
    print(f"  Backend : {'GPU (RAPIDS cuML)' if USE_GPU else 'CPU (scikit-learn)'}")
    print(f"  Started : {timestamp()}")
    print("=" * 72)

    # =================================================================
    # 1. Load & preprocess
    # =================================================================
    print_header("1. LOADING & PREPROCESSING")

    with Timer("Load CSV"):
        df = pd.read_csv(RAW_DATASET_PATH)
        print(f"    Rows: {df.shape[0]:,}, Cols: {df.shape[1]}")

    with Timer("Clean data"):
        df = df.drop_duplicates().dropna().reset_index(drop=True)
        print(f"    After cleaning: {df.shape[0]:,} rows")

    with Timer("Scale features"):
        feature_df = df[CLUSTERING_FEATURES].copy()
        if USE_GPU:
            scaler = cuStandardScaler()
            X_gpu = scaler.fit_transform(cudf.DataFrame(feature_df))
            X = np.asarray(X_gpu)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(feature_df)
        print(f"    Scaled {X.shape[1]} features ({X.shape[0]:,} samples)")

    # =================================================================
    # 2. K-Means optimal k search
    # =================================================================
    print_header("2. K-MEANS OPTIMAL K SEARCH")

    k_range = list(range(2, 11))
    inertias = []
    sil_scores = []

    for k in k_range:
        with Timer(f"k={k}"):
            labels_k, inertia_k = fit_kmeans(X, k)
            sil_k = compute_silhouette(X, labels_k)
            inertias.append(inertia_k)
            sil_scores.append(sil_k)
            print(f"    Inertia: {inertia_k:>12,.1f}  |  Silhouette: {sil_k:.4f}")

    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"\n  --> Best k = {best_k} (silhouette = {max(sil_scores):.4f})")

    # =================================================================
    # 3. K-Means with best k
    # =================================================================
    print_header(f"3. K-MEANS CLUSTERING (k={best_k})")

    with Timer(f"KMeans fit (k={best_k})"):
        kmeans_labels, km_inertia = fit_kmeans(X, best_k)
        sizes = dict(zip(*np.unique(kmeans_labels, return_counts=True)))
        print(f"    Cluster sizes: {sizes}")

    # =================================================================
    # 4. DBSCAN
    # =================================================================
    print_header("4. DBSCAN CLUSTERING")

    with Timer("DBSCAN fit"):
        dbscan_labels = fit_dbscan(X, eps=1.5, min_samples=10)
        n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = int((dbscan_labels == -1).sum())
        print(f"    Clusters: {n_dbscan}, Noise: {n_noise:,}")

    # =================================================================
    # 5. Hierarchical Clustering
    # =================================================================
    print_header(f"5. HIERARCHICAL CLUSTERING (k={best_k})")

    with Timer("Hierarchical fit"):
        hier_labels = fit_hierarchical(X, n_clusters=best_k)
        sizes_h = dict(zip(*np.unique(hier_labels, return_counts=True)))
        print(f"    Cluster sizes: {sizes_h}")

    # =================================================================
    # 6. Evaluation metrics
    # =================================================================
    print_header("6. EVALUATION METRICS")

    metrics = {}

    with Timer("K-Means metrics"):
        km_sil = compute_silhouette(X, kmeans_labels)
        km_dbi = float(davies_bouldin_score(X, kmeans_labels))
        metrics["kmeans"] = {
            "silhouette_score": km_sil,
            "davies_bouldin_index": km_dbi,
            "inertia": km_inertia,
            "n_clusters": best_k,
        }
        print(f"    Silhouette: {km_sil:.4f}  |  DBI: {km_dbi:.4f}  |  "
              f"Inertia: {km_inertia:,.1f}")

    with Timer("DBSCAN metrics"):
        if n_dbscan >= 2:
            mask = dbscan_labels != -1
            db_sil = compute_silhouette(X[mask], dbscan_labels[mask])
            db_dbi = float(davies_bouldin_score(X[mask], dbscan_labels[mask]))
        else:
            db_sil = -1.0
            db_dbi = float("inf")
        metrics["dbscan"] = {
            "silhouette_score": db_sil,
            "davies_bouldin_index": db_dbi,
            "n_clusters": n_dbscan,
            "n_noise": n_noise,
        }
        print(f"    Silhouette: {db_sil:.4f}  |  DBI: {db_dbi:.4f}")

    with Timer("Hierarchical metrics"):
        h_sil = compute_silhouette(X, hier_labels)
        h_dbi = float(davies_bouldin_score(X, hier_labels))
        metrics["hierarchical"] = {
            "silhouette_score": h_sil,
            "davies_bouldin_index": h_dbi,
            "n_clusters": best_k,
        }
        print(f"    Silhouette: {h_sil:.4f}  |  DBI: {h_dbi:.4f}")

    metrics["k_search"] = {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "best_k": best_k,
    }

    # =================================================================
    # 7. t-SNE embedding (sampled)
    # =================================================================
    print_header("7. t-SNE EMBEDDING")

    tsne_sample_size = 8000 if USE_GPU else 5000
    rng = np.random.RandomState(RANDOM_STATE)
    tsne_indices = np.sort(rng.choice(len(X), tsne_sample_size, replace=False))

    with Timer(f"t-SNE ({tsne_sample_size:,} samples)"):
        tsne_coords = compute_tsne(X[tsne_indices])
        tsne_labels = kmeans_labels[tsne_indices]
        print(f"    Output shape: {tsne_coords.shape}")

    # =================================================================
    # 8. Silhouette per-sample values (sampled)
    # =================================================================
    print_header("8. SILHOUETTE DIAGRAM DATA")

    sil_sample_size = 20000 if USE_GPU else 15000
    sil_indices = np.sort(rng.choice(len(X), sil_sample_size, replace=False))

    with Timer(f"Silhouette samples ({sil_sample_size:,} points)"):
        sil_values = silhouette_samples(X[sil_indices],
                                        kmeans_labels[sil_indices])
        sil_labels = kmeans_labels[sil_indices]
        print(f"    Output shape: {sil_values.shape}")

    # =================================================================
    # 9. Cluster profiles
    # =================================================================
    print_header("9. CLUSTER PROFILES (K-Means)")

    with Timer("Cluster profiling"):
        profile_df = feature_df.copy()
        profile_df["Cluster"] = kmeans_labels
        cluster_profiles = profile_df.groupby("Cluster")[CLUSTERING_FEATURES].mean()
        print(cluster_profiles.round(3).to_string())

    # =================================================================
    # 10. Save results
    # =================================================================
    print_header("10. SAVING RESULTS")

    with Timer("Save .npz"):
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
        fsize = os.path.getsize(CLOUD_RESULTS_PATH) / 1024
        print(f"    -> {CLOUD_RESULTS_PATH} ({fsize:.1f} KB)")

    with Timer("Save .json"):
        with open(CLOUD_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"    -> {CLOUD_METRICS_PATH}")

    # =================================================================
    # Summary
    # =================================================================
    total = time.time() - pipeline_start
    print_header("CLOUD COMPUTE COMPLETE")
    print(f"  Backend      : {'GPU (RAPIDS cuML)' if USE_GPU else 'CPU (scikit-learn)'}")
    print(f"  Total time   : {total:.1f}s ({total / 60:.1f} min)")
    print(f"  Finished at  : {timestamp()}")
    print(f"\n  Output files:")
    print(f"    {CLOUD_RESULTS_PATH}")
    print(f"    {CLOUD_METRICS_PATH}")
    print(f"\n  Next step: copy outputs to local machine, then run:")
    print(f"    python main.py")
    print()


if __name__ == "__main__":
    main()
