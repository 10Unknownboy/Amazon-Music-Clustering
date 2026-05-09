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

import matplotlib
matplotlib.use("Agg")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from datetime import datetime

from src.utils import (
    RAW_DATASET_PATH, CLUSTERING_FEATURES, REFERENCE_COLUMNS, DROP_COLUMNS,
    PROCESSED_DATA_DIR, RANDOM_STATE, CLUSTERED_OUTPUT_PATH,
    CLUSTERED_OUTPUT_V2_PATH, RESULTS_PATH, METRICS_PATH,
    SUMMARY_REPORT_PATH, OUTPUTS_DIR,
    print_header, print_subheader, print_metric,
)
from src.data_preprocessing import (
    load_dataset, clean_dataset, drop_non_clustering_columns, scale_features,
    log_transform_duration, winsorize_outliers, add_genre_family, add_decade,
)
from src.feature_selection import (
    select_clustering_features, compute_correlation_matrix, get_feature_summary,
)
from src.evaluation import profile_clusters, interpret_clusters
from src.visualization import (
    plot_feature_distributions_eda, plot_correlation_heatmap, plot_boxplots,
    plot_elbow_curve, plot_silhouette_scores, plot_silhouette_diagram_precomputed,
    plot_pca_clusters, plot_tsne_precomputed, plot_cluster_heatmap,
    plot_cluster_radar, plot_cluster_bar_comparison,
    plot_feature_distributions_by_cluster, plot_cluster_sizes,
    plot_dendrogram_precomputed, plot_genre_per_cluster,
    plot_genre_cluster_heatmap, plot_popularity_by_cluster,
    plot_decade_by_cluster,
)

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
class Timer:
    """Context manager for timestamped phase tracking."""
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.start = time.time()
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] START  {self.label}")
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] DONE   {self.label} ({self.elapsed:.2f}s)")

def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _fit_k(X, k):
    """Fit MiniBatchKMeans for a single k and return (k, inertia, silhouette)."""
    model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                            batch_size=2048, n_init=3, max_iter=100)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=5000, random_state=RANDOM_STATE)
    return k, float(model.inertia_), sil


# =============================================================================
def main():
    """Run the full clustering pipeline with all 10 fixes applied."""
    pipeline_start = time.time()
    print("=" * 72)
    print("  Amazon Music Clustering Pipeline (v2 — all fixes applied)")
    print(f"  Started : {timestamp()}")
    print("=" * 72)

    # =================================================================
    # 1. LOAD & PREPROCESS (Fixes 1, 2, 3, 9)
    # =================================================================
    print_header("1. LOADING & PREPROCESSING")

    with Timer("Load CSV"):
        df = load_dataset()

    with Timer("Clean data"):
        df = clean_dataset(df)

    with Timer("Genre family mapping (Fix 3)"):
        df = add_genre_family(df)

    with Timer("Decade extraction (Fix 9)"):
        df = add_decade(df)

    with Timer("Prepare features"):
        metadata, features_df = drop_non_clustering_columns(df)
        selected_features = select_clustering_features(features_df)

    with Timer("Log-transform duration_ms (Fix 1)"):
        selected_features = log_transform_duration(selected_features)

    with Timer("Winsorize outliers (Fix 2)"):
        selected_features = winsorize_outliers(selected_features)

    with Timer("Feature summary"):
        feature_summary = get_feature_summary(selected_features)
        print(feature_summary.to_string())
        corr_matrix = compute_correlation_matrix(selected_features)

    with Timer("Scale features"):
        scaled_features, scaler = scale_features(selected_features, method="standard")
        X = scaled_features.values
        print(f"    Scaled {X.shape[1]} features ({X.shape[0]:,} samples)")

    # =================================================================
    # 2. K-MEANS OPTIMAL K SEARCH
    # =================================================================
    print_header("2. K-MEANS OPTIMAL K SEARCH")
    k_range = list(range(2, 11))

    with Timer("Parallel k-search"):
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_fit_k)(X, k) for k in k_range)
        results.sort(key=lambda r: r[0])
        inertias = [r[1] for r in results]
        sil_scores = [r[2] for r in results]
        for k, inertia, sil in results:
            print(f"    k={k:2d}  |  Inertia: {inertia:>12,.1f}  |  Silhouette: {sil:.4f}")

    best_k_auto = k_range[int(np.argmax(sil_scores))]
    print(f"\n  --> Auto-selected k = {best_k_auto} (silhouette = {max(sil_scores):.4f})")

    # =================================================================
    # 3. FIX 4: Compare k=3 vs k=6, pick best
    # =================================================================
    print_header("3. K-MEANS: k=3 vs k=6 COMPARISON (Fix 4)")

    k_candidates = [3, 6]
    km_results = {}

    for k in k_candidates:
        with Timer(f"KMeans fit (k={k})"):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=5, max_iter=300, algorithm="elkan")
            labels_k = km.fit_predict(X)
            sil_k = float(silhouette_score(X, labels_k, sample_size=5000,
                                            random_state=RANDOM_STATE))
            dbi_k = float(davies_bouldin_score(X, labels_k))
            km_results[k] = {
                "labels": labels_k, "model": km,
                "silhouette": sil_k, "dbi": dbi_k,
                "inertia": float(km.inertia_),
            }
            sizes = dict(zip(*np.unique(labels_k, return_counts=True)))
            print(f"    k={k}: Sil={sil_k:.4f}, DBI={dbi_k:.4f}, Sizes={sizes}")

    # Pick the k with higher silhouette
    best_k = max(k_candidates, key=lambda k: km_results[k]["silhouette"])
    print(f"\n  --> Selected k = {best_k} (silhouette = {km_results[best_k]['silhouette']:.4f})")

    kmeans_labels = km_results[best_k]["labels"]
    km_inertia = km_results[best_k]["inertia"]
    km_sil = km_results[best_k]["silhouette"]
    km_dbi = km_results[best_k]["dbi"]

    # =================================================================
    # 4. DBSCAN (sampled + propagated)
    # =================================================================
    print_header("4. DBSCAN CLUSTERING")

    with Timer("DBSCAN fit (sampled + propagate)"):
        dbscan_sample_size = min(15000, len(X))
        dbscan_rng = np.random.RandomState(RANDOM_STATE)
        dbscan_idx = dbscan_rng.choice(len(X), dbscan_sample_size, replace=False)
        X_dbscan_sample = X[dbscan_idx]

        dbscan = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
        sample_db_labels = dbscan.fit_predict(X_dbscan_sample)

        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree",
                              leaf_size=50, n_jobs=-1)
        nn.fit(X_dbscan_sample)
        _, nn_indices = nn.kneighbors(X)
        dbscan_labels = sample_db_labels[nn_indices.ravel()]

        n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = int((dbscan_labels == -1).sum())
        print(f"    Clusters: {n_dbscan}, Noise: {n_noise:,}")

    # =================================================================
    # 5. HIERARCHICAL CLUSTERING
    # =================================================================
    print_header(f"5. HIERARCHICAL CLUSTERING (k={best_k})")

    with Timer("Hierarchical fit (sampled + assign)"):
        hier_sample_size = min(8000, len(X))
        hier_rng = np.random.RandomState(RANDOM_STATE)
        hier_idx = hier_rng.choice(len(X), hier_sample_size, replace=False)
        X_hier_sample = X[hier_idx]

        hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        sample_labels = hier.fit_predict(X_hier_sample)

        centroids = np.array([X_hier_sample[sample_labels == c].mean(axis=0)
                              for c in range(best_k)])
        hier_labels = cdist(X, centroids, metric="euclidean").argmin(axis=1)
        print(f"    Cluster sizes: {dict(zip(*np.unique(hier_labels, return_counts=True)))}")

    # =================================================================
    # 6. EVALUATION METRICS
    # =================================================================
    print_header("6. EVALUATION METRICS")
    metrics = {}

    with Timer("K-Means metrics"):
        metrics["kmeans"] = {"silhouette_score": km_sil, "davies_bouldin_index": km_dbi,
                             "inertia": km_inertia, "n_clusters": best_k}
        print(f"    Sil: {km_sil:.4f}  |  DBI: {km_dbi:.4f}  |  Inertia: {km_inertia:,.1f}")

    with Timer("DBSCAN metrics"):
        if n_dbscan >= 2:
            mask = dbscan_labels != -1
            db_sil = float(silhouette_score(X[mask], dbscan_labels[mask],
                                             sample_size=5000, random_state=RANDOM_STATE))
            db_dbi = float(davies_bouldin_score(X[mask], dbscan_labels[mask]))
        else:
            db_sil, db_dbi = -1.0, float("inf")
        metrics["dbscan"] = {"silhouette_score": db_sil, "davies_bouldin_index": db_dbi,
                             "n_clusters": n_dbscan, "n_noise": n_noise}
        print(f"    Sil: {db_sil:.4f}  |  DBI: {db_dbi:.4f}")

    with Timer("Hierarchical metrics"):
        h_sil = float(silhouette_score(X, hier_labels, sample_size=5000,
                                        random_state=RANDOM_STATE))
        h_dbi = float(davies_bouldin_score(X, hier_labels))
        metrics["hierarchical"] = {"silhouette_score": h_sil, "davies_bouldin_index": h_dbi,
                                   "n_clusters": best_k}
        print(f"    Sil: {h_sil:.4f}  |  DBI: {h_dbi:.4f}")

    metrics["k_search"] = {"k_range": k_range, "inertias": inertias,
                           "silhouette_scores": sil_scores, "best_k": best_k}
    metrics["k_comparison"] = {str(k): {"silhouette": v["silhouette"], "dbi": v["dbi"],
                                         "inertia": v["inertia"]}
                                for k, v in km_results.items()}

    comparison = pd.DataFrame({"K-Means": metrics["kmeans"], "DBSCAN": metrics["dbscan"],
                                "Hierarchical": metrics["hierarchical"]}).T
    print_subheader("Algorithm Comparison")
    print(comparison.to_string())

    # =================================================================
    # 7. t-SNE & SILHOUETTE DIAGRAM
    # =================================================================
    print_header("7. t-SNE & SILHOUETTE")
    rng = np.random.RandomState(RANDOM_STATE)

    with Timer("t-SNE (3,000 samples)"):
        tsne_idx = np.sort(rng.choice(len(X), 3000, replace=False))
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                     perplexity=30, max_iter=500, learning_rate="auto")
        tsne_coords = tsne.fit_transform(X[tsne_idx])
        tsne_labels = kmeans_labels[tsne_idx]

    with Timer("Silhouette samples (6,000)"):
        sil_idx = np.sort(rng.choice(len(X), 6000, replace=False))
        sil_values = silhouette_samples(X[sil_idx], kmeans_labels[sil_idx])
        sil_labels = kmeans_labels[sil_idx]

    # =================================================================
    # 8. CLUSTER PROFILES & NAMING (Fix 5)
    # =================================================================
    print_header("8. CLUSTER PROFILES & NAMING (Fix 5)")

    with Timer("Cluster profiling"):
        cluster_profiles = profile_clusters(selected_features, kmeans_labels,
                                            CLUSTERING_FEATURES)
        interpretations = interpret_clusters(cluster_profiles)

    # =================================================================
    # 9. SAVE RESULTS
    # =================================================================
    print_header("9. SAVING RESULTS")

    with Timer("Save .npz"):
        np.savez_compressed(RESULTS_PATH, kmeans_labels=kmeans_labels,
                            dbscan_labels=dbscan_labels, hier_labels=hier_labels,
                            tsne_coords=tsne_coords, tsne_labels=tsne_labels,
                            sil_values=sil_values, sil_labels=sil_labels,
                            scaled_X=X)
        print(f"    -> {RESULTS_PATH} ({os.path.getsize(RESULTS_PATH)/1024:.1f} KB)")

    with Timer("Save .json"):
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"    -> {METRICS_PATH}")

    # =================================================================
    # 10. VISUALIZATIONS
    # =================================================================
    print_header("10. GENERATING VISUALIZATIONS")

    plot_feature_distributions_eda(selected_features, CLUSTERING_FEATURES)
    plot_correlation_heatmap(corr_matrix)
    plot_boxplots(selected_features, CLUSTERING_FEATURES)
    plot_elbow_curve(k_range, inertias, best_k)
    plot_silhouette_scores(k_range, sil_scores, best_k)
    plot_silhouette_diagram_precomputed(sil_values, sil_labels, best_k)
    plot_pca_clusters(X, kmeans_labels, "K-Means")
    plot_pca_clusters(X, dbscan_labels, "DBSCAN")
    plot_pca_clusters(X, hier_labels, "Hierarchical")
    plot_tsne_precomputed(tsne_coords, tsne_labels, "K-Means")
    plot_cluster_heatmap(cluster_profiles)
    plot_cluster_radar(cluster_profiles)
    plot_cluster_bar_comparison(cluster_profiles)
    plot_feature_distributions_by_cluster(selected_features, kmeans_labels,
                                          CLUSTERING_FEATURES)
    plot_cluster_sizes(kmeans_labels)
    plot_dendrogram_precomputed(X)

    # =================================================================
    # 11. BUILD FINAL DATAFRAME & NEW ANALYSES (Fixes 3, 7, 8, 9)
    # =================================================================
    print_header("11. FINAL ANALYSIS & NEW VISUALIZATIONS")

    final_df = metadata.copy()
    for col in CLUSTERING_FEATURES:
        final_df[col] = selected_features[col].values

    final_df["cluster_kmeans"] = kmeans_labels
    final_df["cluster_dbscan"] = dbscan_labels
    final_df["cluster_label"] = final_df["cluster_kmeans"].map(interpretations)

    # Add genre_family and decade from original df
    if 'genre_family' in df.columns:
        final_df['genre_family'] = df['genre_family'].values
    if 'decade' in df.columns:
        final_df['decade'] = df['decade'].values

    # Fix 7: Genre validation per cluster
    with Timer("Genre per cluster plots (Fix 7)"):
        plot_genre_per_cluster(final_df, labels_map=interpretations)
        plot_genre_cluster_heatmap(final_df, labels_map=interpretations)

    # Fix 8: Popularity vs cluster
    with Timer("Popularity analysis (Fix 8)"):
        pop_insight = plot_popularity_by_cluster(final_df, labels_map=interpretations)

    # Fix 9: Decade vs cluster
    with Timer("Decade analysis (Fix 9)"):
        decade_insight = plot_decade_by_cluster(final_df, labels_map=interpretations)

    # Sample tracks per cluster
    print_subheader("Sample Tracks per Cluster")
    for cid in sorted(set(kmeans_labels)):
        tracks = final_df[final_df["cluster_kmeans"] == cid]
        label = interpretations.get(cid, "Unknown")
        print(f"\n  Cluster {cid} -- \"{label}\" ({len(tracks):,} tracks)")
        sample = tracks[["name_song", "name_artists", "genres"]].head(5)
        for _, row in sample.iterrows():
            print(f"    *  {row['name_song']} -- {row['name_artists']} [{row['genres']}]")

    # Export CSV v2
    final_df.to_csv(CLUSTERED_OUTPUT_V2_PATH, index=False)
    # Also keep the original path for backward compat with app.py
    final_df.to_csv(CLUSTERED_OUTPUT_PATH, index=False)
    print(f"\n[INFO] Final dataset exported to: {CLUSTERED_OUTPUT_V2_PATH}")
    print(f"       Total rows: {len(final_df):,}")

    # =================================================================
    # 12. SUMMARY REPORT (Fix 6 — DBSCAN recommendation)
    # =================================================================
    print_header("12. SUMMARY REPORT")

    r = []
    r.append("=" * 72)
    r.append("  AMAZON MUSIC CLUSTERING — SUMMARY REPORT (v2)")
    r.append("=" * 72)
    r.append(f"\n  Generated : {timestamp()}")
    r.append(f"  Dataset   : {RAW_DATASET_PATH}")
    r.append(f"  Songs     : {len(final_df):,}")
    r.append(f"  Features  : {', '.join(CLUSTERING_FEATURES)}")
    r.append(f"\n  Preprocessing applied:")
    r.append(f"    - Log-transformed duration_ms (Fix 1)")
    r.append(f"    - Winsorized speechiness & instrumentalness at p95 (Fix 2)")
    r.append(f"    - Genre families collapsed to ~15 categories (Fix 3)")
    r.append("")

    r.append("=" * 72)
    r.append("  K-MEANS: k=3 vs k=6 COMPARISON (Fix 4)")
    r.append("=" * 72)
    for k in k_candidates:
        v = km_results[k]
        r.append(f"  k={k}: Silhouette={v['silhouette']:.4f}, "
                 f"DBI={v['dbi']:.4f}, Inertia={v['inertia']:,.1f}")
    r.append(f"  --> Selected: k={best_k}")
    r.append("")

    r.append("=" * 72)
    r.append("  ALGORITHM COMPARISON & RECOMMENDATION (Fix 6)")
    r.append("=" * 72)
    r.append(comparison.to_string())
    r.append("")
    r.append("  RECOMMENDATION:")
    if db_sil > km_sil and n_dbscan >= 2:
        r.append(f"    DBSCAN produces higher-quality, more compact clusters")
        r.append(f"    (Silhouette {db_sil:.4f} vs K-Means {km_sil:.4f})")
        r.append(f"    (DBI {db_dbi:.4f} vs K-Means {km_dbi:.4f})")
        r.append(f"    Tradeoff: only {n_dbscan} clusters, {n_noise:,} noise points (~{n_noise/len(X)*100:.1f}%)")
        r.append(f"    Noise = genre-ambiguous / outlier tracks")
    else:
        r.append(f"    K-Means with k={best_k} is recommended for this dataset")
    r.append("")

    r.append("=" * 72)
    r.append("  CLUSTER CHARACTERISTICS (Fix 5 — auto-named)")
    r.append("=" * 72)
    for cid in sorted(interpretations.keys()):
        label = interpretations[cid]
        count = int((kmeans_labels == cid).sum())
        pct = count / len(final_df) * 100
        r.append(f"\n  Cluster {cid}: \"{label}\"")
        r.append(f"    Size: {count:,} songs ({pct:.1f}%)")
        if cid in cluster_profiles.index:
            row = cluster_profiles.loc[cid]
            for feat in CLUSTERING_FEATURES:
                if feat in row.index:
                    r.append(f"    {feat:20s}: {row[feat]:.4f}")
    r.append("")

    # Genre breakdown per cluster (Fix 7)
    r.append("=" * 72)
    r.append("  GENRE FAMILY BREAKDOWN PER CLUSTER (Fix 7)")
    r.append("=" * 72)
    if 'genre_family' in final_df.columns:
        ct = pd.crosstab(final_df['cluster_kmeans'], final_df['genre_family'],
                         normalize='index') * 100
        top5 = ct.mean().sort_values(ascending=False).head(5).index
        for cid in sorted(ct.index):
            label = interpretations.get(cid, "")
            r.append(f"\n  Cluster {cid} (\"{label}\"):")
            for g in top5:
                r.append(f"    {g:20s}: {ct.loc[cid, g]:5.1f}%")
    r.append("")

    # Popularity insight (Fix 8)
    r.append("=" * 72)
    r.append("  POPULARITY INSIGHT (Fix 8)")
    r.append("=" * 72)
    r.append(f"  {pop_insight}")
    r.append("")

    # Decade insight (Fix 9)
    r.append("=" * 72)
    r.append("  DECADE ANALYSIS (Fix 9)")
    r.append("=" * 72)
    r.append(f"  {decade_insight}")
    r.append("")

    r.append("=" * 72)
    r.append("  DELIVERABLES")
    r.append("=" * 72)
    r.append(f"  Clustered CSV : {CLUSTERED_OUTPUT_V2_PATH}")
    r.append(f"  Metrics JSON  : {METRICS_PATH}")
    r.append(f"  Results NPZ   : {RESULTS_PATH}")
    r.append(f"  Summary Report: {SUMMARY_REPORT_PATH}")
    r.append(f"  Plots (18+)   : outputs/plots/")
    r.append(f"  Streamlit App : streamlit run app.py")
    r.append("")
    r.append("=" * 72)
    r.append("  END OF REPORT")
    r.append("=" * 72)

    report_text = "\n".join(r)
    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Report saved to: {SUMMARY_REPORT_PATH}")
    print(report_text)

    total = time.time() - pipeline_start
    print_header("PIPELINE COMPLETE")
    print(f"  Total time : {total:.1f}s ({total/60:.1f} min)")
    return final_df


if __name__ == "__main__":
    main()
