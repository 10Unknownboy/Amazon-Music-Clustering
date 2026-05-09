# =============================================================================
# main.py -- Amazon Music Clustering Pipeline (All-in-One, CPU)
# =============================================================================
# Single-file pipeline that handles everything:
#   1. Load & clean data
#   2. Feature selection & scaling
#   3. K-Means (optimal k search), DBSCAN, Hierarchical clustering
#   4. Evaluation metrics
#   5. t-SNE embedding
#   6. All visualizations
#   7. Final CSV export
#   8. Summary report export
#
# Optimized for speed on CPU:
#   - MiniBatchKMeans for the k-search sweep
#   - Parallel silhouette scoring via joblib
#   - Reduced sample sizes for t-SNE & silhouette diagrams
#   - Aggressive downsampling for expensive O(n^2) operations
#   - BallTree-accelerated DBSCAN for fast density clustering
#
# USAGE:  python main.py
# =============================================================================

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd

# Set matplotlib backend BEFORE any pyplot imports
import matplotlib
matplotlib.use("Agg")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from joblib import Parallel, delayed
from datetime import datetime

from src.utils import (
    RAW_DATASET_PATH,
    CLUSTERING_FEATURES,
    REFERENCE_COLUMNS,
    DROP_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    CLUSTERED_OUTPUT_PATH,
    OUTPUTS_DIR,
    print_header,
    print_subheader,
    print_metric,
)
from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    drop_non_clustering_columns,
    scale_features,
)
from src.feature_selection import (
    select_clustering_features,
    compute_correlation_matrix,
    get_feature_summary,
)
from src.evaluation import (
    profile_clusters,
    interpret_clusters,
)
from src.visualization import (
    plot_feature_distributions_eda,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_silhouette_diagram_precomputed,
    plot_pca_clusters,
    plot_tsne_precomputed,
    plot_cluster_heatmap,
    plot_cluster_radar,
    plot_cluster_bar_comparison,
    plot_feature_distributions_by_cluster,
    plot_cluster_sizes,
    plot_dendrogram_precomputed,
)

# Output paths for cached results
CLOUD_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_results.npz")
CLOUD_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_metrics.json")
SUMMARY_REPORT_PATH = os.path.join(OUTPUTS_DIR, "cluster_summary_report.txt")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Timing utility
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
# Fast clustering helpers
# ---------------------------------------------------------------------------
def _fit_k(X, k):
    """Fit MiniBatchKMeans for a single k and return (k, inertia, silhouette)."""
    model = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_STATE,
        batch_size=2048, n_init=3, max_iter=100,
    )
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=5000,
                           random_state=RANDOM_STATE)
    return k, float(model.inertia_), sil


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the full clustering pipeline on CPU."""
    pipeline_start = time.time()
    print("=" * 72)
    print("  Amazon Music Clustering Pipeline")
    print(f"  Backend : CPU (scikit-learn)")
    print(f"  Started : {timestamp()}")
    print("=" * 72)

    # =================================================================
    # 1. Load & preprocess
    # =================================================================
    print_header("1. LOADING & PREPROCESSING")

    with Timer("Load CSV"):
        df = load_dataset()

    with Timer("Clean data"):
        df = clean_dataset(df)

    with Timer("Prepare features"):
        metadata, features_df = drop_non_clustering_columns(df)
        selected_features = select_clustering_features(features_df)

        feature_summary = get_feature_summary(selected_features)
        print_subheader("Feature Summary")
        print(feature_summary.to_string())

        corr_matrix = compute_correlation_matrix(selected_features)

    with Timer("Scale features"):
        scaled_features, scaler = scale_features(selected_features, method="standard")
        X = scaled_features.values
        print(f"    Scaled {X.shape[1]} features ({X.shape[0]:,} samples)")

    # =================================================================
    # 2. K-Means optimal k search (parallel, MiniBatchKMeans)
    # =================================================================
    print_header("2. K-MEANS OPTIMAL K SEARCH")

    k_range = list(range(2, 11))

    with Timer("Parallel k-search (MiniBatchKMeans)"):
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_fit_k)(X, k) for k in k_range
        )
        # Sort by k to maintain order
        results.sort(key=lambda r: r[0])

        inertias = [r[1] for r in results]
        sil_scores = [r[2] for r in results]

        for k, inertia, sil in results:
            print(f"    k={k:2d}  |  Inertia: {inertia:>12,.1f}  |  "
                  f"Silhouette: {sil:.4f}")

    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"\n  --> Best k = {best_k} (silhouette = {max(sil_scores):.4f})")

    # =================================================================
    # 3. K-Means with best k (full KMeans for final labels)
    # =================================================================
    print_header(f"3. K-MEANS CLUSTERING (k={best_k})")

    with Timer(f"KMeans fit (k={best_k})"):
        kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                        n_init=5, max_iter=300, algorithm="elkan")
        kmeans_labels = kmeans.fit_predict(X)
        km_inertia = float(kmeans.inertia_)
        sizes = dict(zip(*np.unique(kmeans_labels, return_counts=True)))
        print(f"    Cluster sizes: {sizes}")

    # =================================================================
    # 4. DBSCAN (sampled + propagated for speed)
    # =================================================================
    print_header("4. DBSCAN CLUSTERING")

    with Timer("DBSCAN fit (sampled + propagate)"):
        # Full DBSCAN on 95K rows is O(n^2) ≈ 120s.
        # Instead: fit on a 15K subsample, then propagate labels
        # to remaining rows via nearest-neighbor lookup.
        # min_samples is scaled proportionally: 10 * (15000/95837) ≈ 2
        from sklearn.neighbors import NearestNeighbors

        dbscan_sample_size = min(15000, len(X))
        dbscan_rng = np.random.RandomState(RANDOM_STATE)
        dbscan_idx = dbscan_rng.choice(len(X), dbscan_sample_size, replace=False)
        X_dbscan_sample = X[dbscan_idx]

        scaled_min_samples = 5  # Tuned for 15K subsample density
        dbscan = DBSCAN(eps=1.5, min_samples=scaled_min_samples, n_jobs=-1)
        sample_db_labels = dbscan.fit_predict(X_dbscan_sample)

        # Propagate: for each remaining point, assign label of nearest
        # core/non-noise sample point (or -1 if nearest is noise)
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree",
                              leaf_size=50, n_jobs=-1)
        nn.fit(X_dbscan_sample)
        _, nn_indices = nn.kneighbors(X)
        dbscan_labels = sample_db_labels[nn_indices.ravel()]

        n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = int((dbscan_labels == -1).sum())
        print(f"    Sample: {dbscan_sample_size:,} pts (min_samples={scaled_min_samples})")
        print(f"    Clusters: {n_dbscan}, Noise: {n_noise:,}")

    # =================================================================
    # 5. Hierarchical Clustering
    # =================================================================
    print_header(f"5. HIERARCHICAL CLUSTERING (k={best_k})")

    with Timer("Hierarchical fit (sampled + assign)"):
        # Ward linkage needs O(n^2) memory for pairwise distances.
        # Subsample to fit, then assign full dataset via nearest centroid.
        hier_sample_size = min(8000, len(X))
        hier_rng = np.random.RandomState(RANDOM_STATE)
        hier_idx = hier_rng.choice(len(X), hier_sample_size, replace=False)
        X_hier_sample = X[hier_idx]

        hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        sample_labels = hier.fit_predict(X_hier_sample)

        # Compute centroids from sample, assign all rows to nearest centroid
        from scipy.spatial.distance import cdist
        centroids = np.array([
            X_hier_sample[sample_labels == c].mean(axis=0)
            for c in range(best_k)
        ])
        hier_labels = cdist(X, centroids, metric="euclidean").argmin(axis=1)

        sizes_h = dict(zip(*np.unique(hier_labels, return_counts=True)))
        print(f"    Sample: {hier_sample_size:,} pts -> assign all {len(X):,}")
        print(f"    Cluster sizes: {sizes_h}")

    # =================================================================
    # 6. Evaluation metrics
    # =================================================================
    print_header("6. EVALUATION METRICS")

    metrics = {}

    with Timer("K-Means metrics"):
        km_sil = float(silhouette_score(X, kmeans_labels,
                                         sample_size=5000,
                                         random_state=RANDOM_STATE))
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
            db_sil = float(silhouette_score(X[mask], dbscan_labels[mask],
                                             sample_size=5000,
                                             random_state=RANDOM_STATE))
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
        h_sil = float(silhouette_score(X, hier_labels,
                                        sample_size=5000,
                                        random_state=RANDOM_STATE))
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

    # Algorithm comparison table
    print_subheader("Algorithm Comparison")
    comparison = pd.DataFrame({
        "K-Means": metrics["kmeans"],
        "DBSCAN": metrics["dbscan"],
        "Hierarchical": metrics["hierarchical"],
    }).T
    print(comparison.to_string())

    # =================================================================
    # 7. t-SNE embedding (sampled for speed)
    # =================================================================
    print_header("7. t-SNE EMBEDDING")

    tsne_sample_size = 3000
    rng = np.random.RandomState(RANDOM_STATE)
    tsne_indices = np.sort(rng.choice(len(X), tsne_sample_size, replace=False))

    with Timer(f"t-SNE ({tsne_sample_size:,} samples)"):
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                     perplexity=30, max_iter=500, learning_rate="auto")
        tsne_coords = tsne.fit_transform(X[tsne_indices])
        tsne_labels = kmeans_labels[tsne_indices]
        print(f"    Output shape: {tsne_coords.shape}")

    # =================================================================
    # 8. Silhouette per-sample values (sampled)
    # =================================================================
    print_header("8. SILHOUETTE DIAGRAM DATA")

    sil_sample_size = 6000
    sil_indices = np.sort(rng.choice(len(X), sil_sample_size, replace=False))

    with Timer(f"Silhouette samples ({sil_sample_size:,} points)"):
        sil_values = silhouette_samples(X[sil_indices],
                                        kmeans_labels[sil_indices])
        sil_labels = kmeans_labels[sil_indices]
        print(f"    Output shape: {sil_values.shape}")

    # =================================================================
    # 9. Cluster profiles & interpretation
    # =================================================================
    print_header("9. CLUSTER PROFILES (K-Means)")

    with Timer("Cluster profiling"):
        cluster_profiles = profile_clusters(
            selected_features, kmeans_labels, CLUSTERING_FEATURES
        )
        interpretations = interpret_clusters(cluster_profiles)

    # =================================================================
    # 10. Save computed results (for caching / reuse)
    # =================================================================
    print_header("10. SAVING COMPUTED RESULTS")

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
    # 11. Generate all visualizations
    # =================================================================
    print_header("11. GENERATING VISUALIZATIONS")

    # EDA plots
    plot_feature_distributions_eda(selected_features, CLUSTERING_FEATURES)
    plot_correlation_heatmap(corr_matrix)
    plot_boxplots(selected_features, CLUSTERING_FEATURES)

    # Elbow & silhouette score plots
    plot_elbow_curve(k_range, inertias, best_k)
    plot_silhouette_scores(k_range, sil_scores, best_k)

    # Silhouette diagram
    plot_silhouette_diagram_precomputed(sil_values, sil_labels, best_k)

    # PCA scatter plots (PCA is fast)
    plot_pca_clusters(X, kmeans_labels, "K-Means")
    plot_pca_clusters(X, dbscan_labels, "DBSCAN")
    plot_pca_clusters(X, hier_labels, "Hierarchical")

    # t-SNE scatter (from computed embedding)
    plot_tsne_precomputed(tsne_coords, tsne_labels, "K-Means")

    # Cluster profile plots
    plot_cluster_heatmap(cluster_profiles)
    plot_cluster_radar(cluster_profiles)
    plot_cluster_bar_comparison(cluster_profiles)
    plot_feature_distributions_by_cluster(
        selected_features, kmeans_labels, CLUSTERING_FEATURES
    )
    plot_cluster_sizes(kmeans_labels)

    # Dendrogram (small sample)
    plot_dendrogram_precomputed(X)

    # =================================================================
    # 12. Final analysis & export
    # =================================================================
    print_header("12. FINAL ANALYSIS & EXPORT")

    # Build final DataFrame
    final_df = metadata.copy()
    for col in CLUSTERING_FEATURES:
        final_df[col] = selected_features[col].values

    final_df["cluster_kmeans"] = kmeans_labels
    final_df["cluster_dbscan"] = dbscan_labels
    final_df["cluster_hierarchical"] = hier_labels
    final_df["cluster_label"] = final_df["cluster_kmeans"].map(interpretations)

    # Show sample tracks per cluster
    print_subheader("Sample Tracks per Cluster (K-Means)")
    for cid in sorted(set(kmeans_labels)):
        tracks = final_df[final_df["cluster_kmeans"] == cid]
        label = interpretations.get(cid, "Unknown")
        print(f"\n  Cluster {cid} -- \"{label}\" ({len(tracks):,} tracks)")
        sample = tracks[["name_song", "name_artists", "genres"]].head(5)
        for _, row in sample.iterrows():
            print(f"    *  {row['name_song']} -- {row['name_artists']} "
                  f"[{row['genres']}]")

    # Export CSV
    final_df.to_csv(CLUSTERED_OUTPUT_PATH, index=False)
    print(f"\n[INFO] Final dataset exported to: {CLUSTERED_OUTPUT_PATH}")
    print(f"       Total rows: {len(final_df):,}")

    # =================================================================
    # 13. Summary report
    # =================================================================
    print_header("13. SUMMARY REPORT")

    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("  AMAZON MUSIC CLUSTERING — SUMMARY REPORT")
    report_lines.append("=" * 72)
    report_lines.append(f"\n  Generated : {timestamp()}")
    report_lines.append(f"  Dataset   : {RAW_DATASET_PATH}")
    report_lines.append(f"  Total Songs : {len(final_df):,}")
    report_lines.append(f"  Features  : {len(CLUSTERING_FEATURES)} audio features")
    report_lines.append(f"  Features  : {', '.join(CLUSTERING_FEATURES)}")
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  OPTIMAL CLUSTERING CONFIGURATION")
    report_lines.append("=" * 72)
    report_lines.append(f"  Algorithm       : K-Means")
    report_lines.append(f"  Optimal k       : {best_k}")
    report_lines.append(f"  Silhouette Score: {km_sil:.4f}")
    report_lines.append(f"  Davies-Bouldin  : {km_dbi:.4f}")
    report_lines.append(f"  Inertia (SSE)   : {km_inertia:,.1f}")
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  ALGORITHM COMPARISON")
    report_lines.append("=" * 72)
    report_lines.append(comparison.to_string())
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  CLUSTER CHARACTERISTICS (K-Means)")
    report_lines.append("=" * 72)
    for cid in sorted(interpretations.keys()):
        label = interpretations[cid]
        count = int((kmeans_labels == cid).sum())
        pct = count / len(final_df) * 100
        report_lines.append(f"\n  Cluster {cid}: \"{label}\"")
        report_lines.append(f"    Size: {count:,} songs ({pct:.1f}%)")
        if cid in cluster_profiles.index:
            row = cluster_profiles.loc[cid]
            for feat in CLUSTERING_FEATURES:
                if feat in row.index:
                    report_lines.append(f"    {feat:20s}: {row[feat]:.4f}")
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  SAMPLE TRACKS PER CLUSTER")
    report_lines.append("=" * 72)
    for cid in sorted(set(kmeans_labels)):
        tracks = final_df[final_df["cluster_kmeans"] == cid]
        label = interpretations.get(cid, "Unknown")
        report_lines.append(f"\n  Cluster {cid} — \"{label}\"")
        sample = tracks[["name_song", "name_artists", "genres"]].head(5)
        for _, row in sample.iterrows():
            report_lines.append(
                f"    • {row['name_song']} — {row['name_artists']} "
                f"[{row['genres']}]"
            )
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  VISUALIZATIONS GENERATED")
    report_lines.append("=" * 72)
    report_lines.append("  All plots saved to: outputs/plots/")
    report_lines.append("  01. Feature Distributions (Raw Data)")
    report_lines.append("  02. Feature Correlation Heatmap")
    report_lines.append("  03. Feature Boxplots (Normalized)")
    report_lines.append("  04. Elbow Curve (SSE vs. k)")
    report_lines.append("  05. Silhouette Scores (k comparison)")
    report_lines.append("  06. Silhouette Diagram (per-sample)")
    report_lines.append("  07. PCA Cluster Scatter (K-Means, DBSCAN, Hierarchical)")
    report_lines.append("  08. t-SNE Cluster Visualization")
    report_lines.append("  09. Cluster Feature Heatmap")
    report_lines.append("  10. Cluster Radar Profiles")
    report_lines.append("  11. Feature Bar Comparison Across Clusters")
    report_lines.append("  12. Feature Distributions by Cluster")
    report_lines.append("  13. Cluster Size Distribution")
    report_lines.append("  14. Hierarchical Clustering Dendrogram")
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  PROJECT DELIVERABLES")
    report_lines.append("=" * 72)
    report_lines.append(f"  Clustered CSV : {CLUSTERED_OUTPUT_PATH}")
    report_lines.append(f"  Metrics JSON  : {CLOUD_METRICS_PATH}")
    report_lines.append(f"  Results NPZ   : {CLOUD_RESULTS_PATH}")
    report_lines.append(f"  Summary Report: {SUMMARY_REPORT_PATH}")
    report_lines.append(f"  Streamlit App : streamlit run app.py")
    report_lines.append("")
    report_lines.append("=" * 72)
    report_lines.append("  END OF REPORT")
    report_lines.append("=" * 72)

    report_text = "\n".join(report_lines)

    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Report saved to: {SUMMARY_REPORT_PATH}")
    print(report_text)

    # =================================================================
    # Summary
    # =================================================================
    total = time.time() - pipeline_start
    print_header("PIPELINE COMPLETE")
    print(f"  Dataset          : {len(final_df):,} songs")
    print(f"  Features used    : {len(CLUSTERING_FEATURES)}")
    print(f"  Optimal K        : {best_k}")
    print(f"  Silhouette (KM)  : {km_sil:.4f}")
    print(f"  Davies-Bouldin   : {km_dbi:.4f}")
    print(f"  Output CSV       : {CLUSTERED_OUTPUT_PATH}")
    print(f"  Summary Report   : {SUMMARY_REPORT_PATH}")
    print(f"  Plots saved to   : outputs/plots/")
    print(f"  Total time       : {total:.1f}s ({total / 60:.1f} min)")
    print()

    return final_df


if __name__ == "__main__":
    main()
