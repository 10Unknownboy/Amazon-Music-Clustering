# =============================================================================
# main.py -- Amazon Music Clustering Pipeline (Lightweight Local Runner)
# =============================================================================
# This script performs ZERO heavy ML computation. It loads pre-computed
# results from cloud_compute.py and:
#   1. Loads raw data (for metadata)
#   2. Preprocesses features (fast: just scaling)
#   3. Loads cloud-computed cluster labels, metrics, and embeddings
#   4. Generates ALL 14 visualization plots
#   5. Exports the final CSV with cluster labels
#
# PREREQUISITES:
#   Run cloud_compute.py on a cloud server first, then copy:
#     data/processed/cloud_results.npz
#     data/processed/cloud_metrics.json
#   to this project before running main.py.
#
# EXPECTED RUNTIME: < 30 seconds on any machine.
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

from src.utils import (
    print_header,
    print_subheader,
    CLUSTERED_OUTPUT_PATH,
    CLUSTERING_FEATURES,
    REFERENCE_COLUMNS,
    DROP_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_DATASET_PATH,
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

# Paths for cloud-computed results
CLOUD_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_results.npz")
CLOUD_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "cloud_metrics.json")


def load_cloud_results():
    """
    Load pre-computed results from cloud_compute.py.

    Returns
    -------
    tuple of (dict, dict)
        - arrays: dict of numpy arrays (labels, embeddings, etc.)
        - metrics: dict of evaluation metrics and k-search data
    """
    if not os.path.exists(CLOUD_RESULTS_PATH):
        raise FileNotFoundError(
            f"Cloud results not found at: {CLOUD_RESULTS_PATH}\n"
            f"Run 'python cloud_compute.py' on a cloud server first, "
            f"then copy the output files here."
        )

    arrays = dict(np.load(CLOUD_RESULTS_PATH, allow_pickle=True))
    with open(CLOUD_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    print(f"[INFO] Loaded cloud results from: {CLOUD_RESULTS_PATH}")
    print(f"       Arrays: {list(arrays.keys())}")
    print(f"       Metrics: {list(metrics.keys())}")
    return arrays, metrics


def main():
    """Run the lightweight local pipeline."""
    pipeline_start = time.time()

    # =================================================================
    # PHASE 1: Load raw data
    # =================================================================
    print_header("PHASE 1: DATA LOADING")

    df = load_dataset()

    # =================================================================
    # PHASE 2: Preprocessing (fast -- just cleaning & scaling)
    # =================================================================
    print_header("PHASE 2: PREPROCESSING")

    df = clean_dataset(df)
    metadata, features_df = drop_non_clustering_columns(df)
    selected_features = select_clustering_features(features_df)

    feature_summary = get_feature_summary(selected_features)
    print_subheader("Feature Summary")
    print(feature_summary.to_string())

    corr_matrix = compute_correlation_matrix(selected_features)
    scaled_features, scaler = scale_features(selected_features, method="standard")
    X = scaled_features.values

    # =================================================================
    # PHASE 3: Load cloud-computed results
    # =================================================================
    print_header("PHASE 3: LOADING CLOUD-COMPUTED RESULTS")

    cloud_arrays, cloud_metrics = load_cloud_results()

    kmeans_labels = cloud_arrays["kmeans_labels"]
    dbscan_labels = cloud_arrays["dbscan_labels"]
    hier_labels = cloud_arrays["hier_labels"]
    tsne_coords = cloud_arrays["tsne_coords"]
    tsne_labels = cloud_arrays["tsne_labels"]
    sil_values = cloud_arrays["sil_values"]
    sil_labels = cloud_arrays["sil_labels"]

    k_search = cloud_metrics["k_search"]
    km_metrics = cloud_metrics["kmeans"]
    db_metrics = cloud_metrics["dbscan"]
    h_metrics = cloud_metrics["hierarchical"]
    optimal_k = k_search["best_k"]

    # Print pre-computed metrics
    print_subheader("Pre-computed Evaluation Metrics")
    print(f"\n  K-Means (k={optimal_k}):")
    print(f"    Silhouette     : {km_metrics['silhouette_score']:.4f}")
    print(f"    Davies-Bouldin : {km_metrics['davies_bouldin_index']:.4f}")
    print(f"    Inertia        : {km_metrics['inertia']:,.1f}")

    print(f"\n  DBSCAN:")
    print(f"    Clusters       : {db_metrics['n_clusters']}")
    print(f"    Noise points   : {db_metrics['n_noise']:,}")
    print(f"    Silhouette     : {db_metrics['silhouette_score']:.4f}")
    print(f"    Davies-Bouldin : {db_metrics['davies_bouldin_index']:.4f}")

    print(f"\n  Hierarchical (k={optimal_k}):")
    print(f"    Silhouette     : {h_metrics['silhouette_score']:.4f}")
    print(f"    Davies-Bouldin : {h_metrics['davies_bouldin_index']:.4f}")

    # Algorithm comparison table
    print_subheader("Algorithm Comparison")
    comparison = pd.DataFrame({
        "K-Means": km_metrics,
        "DBSCAN": db_metrics,
        "Hierarchical": h_metrics,
    }).T
    print(comparison.to_string())

    # Cluster profiles & interpretation
    cluster_profiles = profile_clusters(
        selected_features, kmeans_labels, CLUSTERING_FEATURES
    )
    interpretations = interpret_clusters(cluster_profiles)

    # =================================================================
    # PHASE 4: Generate ALL visualizations (rendering only -- fast)
    # =================================================================
    print_header("PHASE 4: GENERATING VISUALIZATIONS")

    # --- EDA plots ---
    plot_feature_distributions_eda(selected_features, CLUSTERING_FEATURES)
    plot_correlation_heatmap(corr_matrix)
    plot_boxplots(selected_features, CLUSTERING_FEATURES)

    # --- Elbow & silhouette score plots (from pre-computed data) ---
    plot_elbow_curve(
        k_search["k_range"], k_search["inertias"], optimal_k
    )
    plot_silhouette_scores(
        k_search["k_range"], k_search["silhouette_scores"], optimal_k
    )

    # --- Silhouette diagram (from pre-computed per-sample values) ---
    plot_silhouette_diagram_precomputed(sil_values, sil_labels, optimal_k)

    # --- PCA scatter plots (PCA is fast, computed here) ---
    plot_pca_clusters(X, kmeans_labels, "K-Means")
    plot_pca_clusters(X, dbscan_labels, "DBSCAN")
    plot_pca_clusters(X, hier_labels, "Hierarchical")

    # --- t-SNE scatter (from pre-computed embedding) ---
    plot_tsne_precomputed(tsne_coords, tsne_labels, "K-Means")

    # --- Cluster profile plots ---
    plot_cluster_heatmap(cluster_profiles)
    plot_cluster_radar(cluster_profiles)
    plot_cluster_bar_comparison(cluster_profiles)
    plot_feature_distributions_by_cluster(
        selected_features, kmeans_labels, CLUSTERING_FEATURES
    )
    plot_cluster_sizes(kmeans_labels)

    # --- Dendrogram (small sample, computed here -- fast) ---
    plot_dendrogram_precomputed(X)

    # =================================================================
    # PHASE 5: Final analysis & export
    # =================================================================
    print_header("PHASE 5: FINAL ANALYSIS & EXPORT")

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
    # Summary
    # =================================================================
    elapsed = time.time() - pipeline_start
    print_header("PIPELINE COMPLETE")
    print(f"  Dataset          : {len(final_df):,} songs")
    print(f"  Features used    : {len(CLUSTERING_FEATURES)}")
    print(f"  Optimal K        : {optimal_k}")
    print(f"  Silhouette (KM)  : {km_metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin   : {km_metrics['davies_bouldin_index']:.4f}")
    print(f"  Output CSV       : {CLUSTERED_OUTPUT_PATH}")
    print(f"  Plots saved to   : outputs/plots/")
    print(f"  Total time       : {elapsed:.1f}s")
    print()

    return final_df


if __name__ == "__main__":
    main()
