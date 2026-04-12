# =============================================================================
# main.py -- Amazon Music Clustering Pipeline Orchestrator
# =============================================================================
# End-to-end pipeline that runs all phases of the clustering project:
#   1. Data loading & exploration
#   2. Preprocessing & cleaning
#   3. Feature selection & correlation analysis
#   4. K-Means clustering (with optimal k search)
#   5. DBSCAN clustering
#   6. Hierarchical clustering
#   7. Evaluation & interpretation
#   8. Visualization (all plots)
#   9. Export results to CSV
# =============================================================================

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd

# Set matplotlib backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use("Agg")

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    print_header,
    print_subheader,
    CLUSTERED_OUTPUT_PATH,
    CLUSTERING_FEATURES,
)
from src.data_preprocessing import (
    load_dataset,
    explore_dataset,
    clean_dataset,
    drop_non_clustering_columns,
    scale_features,
)
from src.feature_selection import (
    select_clustering_features,
    compute_correlation_matrix,
    get_feature_summary,
)
from src.clustering import (
    find_optimal_k,
    run_kmeans,
    run_dbscan,
    run_hierarchical,
)
from src.evaluation import (
    evaluate_all,
    profile_clusters,
    interpret_clusters,
)
from src.visualization import (
    plot_feature_distributions_eda,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_silhouette_diagram,
    plot_pca_clusters,
    plot_tsne_clusters,
    plot_cluster_heatmap,
    plot_cluster_radar,
    plot_cluster_bar_comparison,
    plot_feature_distributions_by_cluster,
    plot_cluster_sizes,
    plot_dendrogram,
)


def main():
    """Run the full Amazon Music Clustering pipeline."""
    pipeline_start = time.time()

    # =================================================================
    # PHASE 1: Data Loading & Exploration
    # =================================================================
    print_header("PHASE 1: DATA LOADING & EXPLORATION")

    df = load_dataset()
    explore_dataset(df)

    # =================================================================
    # PHASE 2: Preprocessing & Cleaning
    # =================================================================
    print_header("PHASE 2: PREPROCESSING & CLEANING")

    df = clean_dataset(df)
    metadata, features_df = drop_non_clustering_columns(df)

    # =================================================================
    # PHASE 3: Feature Selection & Correlation Analysis
    # =================================================================
    print_header("PHASE 3: FEATURE SELECTION & CORRELATION ANALYSIS")

    selected_features = select_clustering_features(features_df)
    feature_summary = get_feature_summary(selected_features)
    print_subheader("Feature Summary")
    print(feature_summary.to_string())

    corr_matrix = compute_correlation_matrix(selected_features)

    # Scale features for clustering
    scaled_features, scaler = scale_features(selected_features, method="standard")
    X = scaled_features.values

    # =================================================================
    # PHASE 3.5: EDA Visualizations
    # =================================================================
    print_header("GENERATING EDA VISUALIZATIONS")

    plot_feature_distributions_eda(selected_features, CLUSTERING_FEATURES)
    plot_correlation_heatmap(corr_matrix)
    plot_boxplots(selected_features, CLUSTERING_FEATURES)

    # =================================================================
    # PHASE 4A: K-Means Clustering
    # =================================================================
    print_header("PHASE 4A: K-MEANS CLUSTERING")

    # Find optimal k using Elbow and Silhouette methods
    t0 = time.time()
    k_results = find_optimal_k(X)
    optimal_k = k_results["recommended_k"]
    print(f"  [TIME] Optimal-k search: {time.time() - t0:.1f}s")

    # Plot elbow and silhouette curves
    plot_elbow_curve(k_results["k_range"], k_results["inertias"], optimal_k)
    plot_silhouette_scores(
        k_results["k_range"], k_results["silhouette_scores"], optimal_k
    )

    # Run K-Means with optimal k
    kmeans_labels, kmeans_model = run_kmeans(X, optimal_k)

    # =================================================================
    # PHASE 4B: DBSCAN Clustering
    # =================================================================
    print_header("PHASE 4B: DBSCAN CLUSTERING")

    dbscan_labels, dbscan_model = run_dbscan(X)

    # =================================================================
    # PHASE 4C: Hierarchical Clustering
    # =================================================================
    print_header("PHASE 4C: HIERARCHICAL CLUSTERING")

    hierarchical_labels, hierarchical_model = run_hierarchical(
        X, n_clusters=optimal_k
    )

    # =================================================================
    # PHASE 5: Evaluation & Interpretation (K-Means as primary)
    # =================================================================
    print_header("PHASE 5: CLUSTER EVALUATION & INTERPRETATION")

    # Evaluate all three algorithms
    print_subheader("K-Means Evaluation")
    kmeans_metrics = evaluate_all(X, kmeans_labels, kmeans_model)

    print_subheader("DBSCAN Evaluation")
    dbscan_metrics = evaluate_all(X, dbscan_labels)

    print_subheader("Hierarchical Evaluation")
    hierarchical_metrics = evaluate_all(X, hierarchical_labels)

    # Comparison table
    print_header("ALGORITHM COMPARISON")
    comparison = pd.DataFrame({
        "K-Means": kmeans_metrics,
        "DBSCAN": dbscan_metrics,
        "Hierarchical": hierarchical_metrics,
    }).T
    print(comparison.to_string())

    # Profile K-Means clusters (primary algorithm)
    cluster_profiles = profile_clusters(
        selected_features, kmeans_labels, CLUSTERING_FEATURES
    )
    interpretations = interpret_clusters(cluster_profiles)

    # =================================================================
    # PHASE 6: Visualization
    # =================================================================
    print_header("PHASE 6: GENERATING CLUSTER VISUALIZATIONS")

    # Silhouette diagram for K-Means
    plot_silhouette_diagram(X, kmeans_labels)

    # PCA scatter plots for all algorithms
    plot_pca_clusters(X, kmeans_labels, "K-Means")
    plot_pca_clusters(X, dbscan_labels, "DBSCAN")
    plot_pca_clusters(X, hierarchical_labels, "Hierarchical")

    # t-SNE scatter plot (K-Means only -- sampled for speed)
    t0 = time.time()
    plot_tsne_clusters(X, kmeans_labels, "K-Means")
    print(f"  [TIME] t-SNE: {time.time() - t0:.1f}s")

    # Cluster profile visualizations
    plot_cluster_heatmap(cluster_profiles)
    plot_cluster_radar(cluster_profiles)
    plot_cluster_bar_comparison(cluster_profiles)
    plot_feature_distributions_by_cluster(
        selected_features, kmeans_labels, CLUSTERING_FEATURES
    )
    plot_cluster_sizes(kmeans_labels)

    # Dendrogram
    plot_dendrogram(X)

    # =================================================================
    # PHASE 7: Final Analysis & Export
    # =================================================================
    print_header("PHASE 7: FINAL ANALYSIS & EXPORT")

    # Build final DataFrame with cluster labels
    final_df = metadata.copy()
    for col in CLUSTERING_FEATURES:
        final_df[col] = selected_features[col].values

    final_df["cluster_kmeans"] = kmeans_labels
    final_df["cluster_dbscan"] = dbscan_labels
    final_df["cluster_hierarchical"] = hierarchical_labels

    # Add cluster interpretation labels
    final_df["cluster_label"] = final_df["cluster_kmeans"].map(interpretations)

    # Show top tracks per cluster
    print_subheader("Sample Tracks per Cluster (K-Means)")
    for cluster_id in sorted(set(kmeans_labels)):
        cluster_tracks = final_df[final_df["cluster_kmeans"] == cluster_id]
        label = interpretations.get(cluster_id, "Unknown")
        print(f"\n  Cluster {cluster_id} -- \"{label}\" ({len(cluster_tracks):,} tracks)")
        sample = cluster_tracks[["name_song", "name_artists", "genres"]].head(5)
        for _, row in sample.iterrows():
            print(f"    *  {row['name_song']} -- {row['name_artists']} [{row['genres']}]")

    # Export to CSV
    final_df.to_csv(CLUSTERED_OUTPUT_PATH, index=False)
    print(f"\n[INFO] Final dataset exported to: {CLUSTERED_OUTPUT_PATH}")
    print(f"       Total rows: {len(final_df):,}")
    print(f"       Columns: {list(final_df.columns)}")

    # =================================================================
    # Summary
    # =================================================================
    elapsed = time.time() - pipeline_start
    print_header("PIPELINE COMPLETE")
    print(f"  Dataset          : {len(final_df):,} songs")
    print(f"  Features used    : {len(CLUSTERING_FEATURES)}")
    print(f"  Optimal K        : {optimal_k}")
    print(f"  Silhouette (KM)  : {kmeans_metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin   : {kmeans_metrics['davies_bouldin_index']:.4f}")
    print(f"  Output CSV       : {CLUSTERED_OUTPUT_PATH}")
    print(f"  Plots saved to   : outputs/plots/")
    print(f"  Total time       : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print()

    return final_df


if __name__ == "__main__":
    main()
