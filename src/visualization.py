# =============================================================================
# visualization.py -- Cluster Visualization & Plotting
# =============================================================================
# All plotting functions for the clustering pipeline. Generates publication-
# quality visualizations for EDA, cluster analysis, and evaluation.
# Every plot is saved to outputs/plots/ and optionally displayed.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

from src.utils import PLOTS_DIR, RANDOM_STATE, print_subheader


# ---------------------------------------------------------------------------
# Global plot style configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_LARGE = (12, 8)
DPI = 150

# Custom color palette for clusters
CLUSTER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F1948A", "#82E0AA", "#F8C471", "#AED6F1", "#D7BDE2",
]


def _save_plot(fig, filename):
    """Save a figure to the plots directory."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {filepath}")


def _sample_data(X, labels, sample_size, seed=RANDOM_STATE):
    """
    Downsample X and labels to sample_size rows for expensive computations.
    Returns (X_sample, labels_sample).
    """
    if len(X) <= sample_size:
        X_out = X if isinstance(X, np.ndarray) else X.values
        return X_out, np.array(labels)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), sample_size, replace=False)
    X_out = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices].values
    labels_out = np.array(labels)[indices]
    print(f"  Sampled {sample_size:,} / {len(X):,} points.")
    return X_out, labels_out


# =============================================================================
# EDA Visualizations
# =============================================================================
def plot_feature_distributions_eda(df, features=None):
    """
    Plot histograms for each audio feature.

    Parameters
    ----------
    df : pd.DataFrame
        Unscaled feature DataFrame.
    features : list of str, optional
        Features to plot. Defaults to all columns.
    """
    if features is None:
        features = list(df.columns)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df[feature], bins=50, alpha=0.7,
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                edgecolor="white", linewidth=0.5)
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (Raw Data)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_plot(fig, "01_feature_distributions.png")


def plot_correlation_heatmap(corr_matrix):
    """
    Plot a triangular correlation heatmap of audio features.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Correlation"}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save_plot(fig, "02_correlation_heatmap.png")


def plot_boxplots(df, features=None):
    """
    Plot box plots for each feature (normalized for comparison).

    Parameters
    ----------
    df : pd.DataFrame
        Unscaled feature DataFrame.
    features : list of str, optional
        Features to plot. Defaults to all columns.
    """
    if features is None:
        features = list(df.columns)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    from sklearn.preprocessing import MinMaxScaler
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[features]), columns=features
    )

    scaled.boxplot(ax=ax, grid=False, patch_artist=True,
                   boxprops=dict(facecolor="#4ECDC4", alpha=0.7),
                   medianprops=dict(color="#FF6B6B", linewidth=2))
    ax.set_title("Feature Distributions (Normalized for Comparison)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Value")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save_plot(fig, "03_feature_boxplots.png")


# =============================================================================
# Clustering Evaluation Visualizations
# =============================================================================
def plot_elbow_curve(k_range, inertias, optimal_k=None):
    """
    Plot the Elbow Method curve (SSE vs. number of clusters).

    Parameters
    ----------
    k_range : list of int
        Range of k values tested.
    inertias : list of float
        Inertia (SSE) for each k.
    optimal_k : int, optional
        The optimal k to mark on the plot.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    ax.plot(k_range, inertias, "o-", color="#45B7D1", linewidth=2.5,
            markersize=8, markerfacecolor="#FF6B6B", markeredgecolor="white",
            markeredgewidth=2)

    if optimal_k is not None:
        idx = list(k_range).index(optimal_k)
        ax.axvline(x=optimal_k, color="#FF6B6B", linestyle="--",
                   linewidth=1.5, alpha=0.7)
        ax.scatter([optimal_k], [inertias[idx]], s=200, color="#FF6B6B",
                   zorder=5, edgecolors="white", linewidths=2)
        ax.annotate(f"Optimal k = {optimal_k}",
                    xy=(optimal_k, inertias[idx]),
                    xytext=(optimal_k + 0.5, inertias[idx]),
                    fontsize=12, fontweight="bold", color="#FF6B6B")

    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (SSE)", fontsize=12)
    ax.set_title("Elbow Method -- Optimal Number of Clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(k_range)
    fig.tight_layout()
    _save_plot(fig, "04_elbow_curve.png")


def plot_silhouette_scores(k_range, scores, optimal_k=None):
    """
    Plot silhouette scores for different k values.

    Parameters
    ----------
    k_range : list of int
        Range of k values tested.
    scores : list of float
        Silhouette score for each k.
    optimal_k : int, optional
        The optimal k to highlight.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    colors = ["#FF6B6B" if (optimal_k and k == optimal_k)
              else "#4ECDC4" for k in k_range]
    bars = ax.bar(k_range, scores, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{score:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs. Number of Clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(k_range)
    fig.tight_layout()
    _save_plot(fig, "05_silhouette_scores.png")


def plot_silhouette_diagram(X, labels, sample_size=15000):
    """
    Plot a per-sample silhouette analysis diagram.

    Samples the dataset to keep rendering fast on large data.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    sample_size : int
        Max samples to use (silhouette_samples is O(n^2)).
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print("  [SKIP] Silhouette diagram requires at least 2 clusters.")
        return

    # Downsample for speed — silhouette_samples is expensive
    X_s, labels_s = _sample_data(X, labels, sample_size)

    sample_silhouette_values = silhouette_samples(X_s, labels_s)
    avg_score = np.mean(sample_silhouette_values)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    y_lower = 10
    for i in range(n_clusters):
        cluster_values = sample_silhouette_values[labels_s == i]
        cluster_values.sort()
        size = cluster_values.shape[0]
        y_upper = y_lower + size

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.75)
        ax.text(-0.03, y_lower + 0.5 * size, str(i),
                fontsize=10, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="#FF6B6B", linestyle="--", linewidth=2,
               label=f"Avg Score: {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Silhouette Diagram -- Per-Sample Analysis",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_plot(fig, "06_silhouette_diagram.png")


# =============================================================================
# Cluster Visualizations (PCA & t-SNE)
# =============================================================================
def plot_pca_clusters(X, labels, title_suffix="K-Means"):
    """
    Reduce features to 2D using PCA and plot color-coded clusters.

    For DBSCAN, uses high-contrast colors and enlarged minority cluster
    markers with an info text box showing cluster sizes.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    title_suffix : str
        Label to add to the plot title (e.g., algorithm name).
    """
    print_subheader(f"PCA Scatter Plot -- {title_suffix}")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    is_dbscan = title_suffix.upper() == "DBSCAN"
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    unique_labels = sorted(set(labels))

    # DBSCAN-specific high-contrast palette and marker sizes
    dbscan_colors = {-1: "#AAAAAA", 0: "#4C9BE8", 1: "#E8834C"}
    dbscan_sizes = {-1: 3, 0: 5, 1: 15}

    for label in unique_labels:
        mask = labels == label
        if is_dbscan:
            color = dbscan_colors.get(label, CLUSTER_COLORS[label % len(CLUSTER_COLORS)])
            size = dbscan_sizes.get(label, 5)
            alpha = 0.25 if label == -1 else 0.45
        else:
            color = "#999999" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            size = 5
            alpha = 0.4
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=alpha, s=size, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=12)

    if is_dbscan:
        ax.set_title(
            "PCA Cluster Visualization -- DBSCAN\n"
            "Note: DBSCAN found 2 uneven clusters. Cluster 0 = mainstream songs,\n"
            "Cluster 1 = outlier tracks. Noise points shown in gray.",
            fontsize=12, fontweight="bold")
        # Info text box with cluster sizes
        parts = []
        for lbl in unique_labels:
            n = int((labels == lbl).sum())
            tag = "Noise" if lbl == -1 else f"Cluster {lbl}"
            parts.append(f"{tag}: {n:,} songs")
        ax.text(0.02, 0.97, "  |  ".join(parts), transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="#ccc"))
    else:
        ax.set_title(f"PCA Cluster Visualization -- {title_suffix}",
                     fontsize=14, fontweight="bold")

    ax.legend(markerscale=3, fontsize=10, loc="best")
    fig.tight_layout()

    filename = f"07_pca_clusters_{title_suffix.lower().replace(' ', '_')}.png"
    _save_plot(fig, filename)
    return pca


def plot_tsne_clusters(X, labels, title_suffix="K-Means", sample_size=5000):
    """
    Reduce features to 2D using t-SNE and plot color-coded clusters.

    t-SNE is computationally expensive, so we aggressively sample.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    title_suffix : str
        Algorithm name for the title.
    sample_size : int
        Max number of points to plot (default 5000 for speed).
    """
    print_subheader(f"t-SNE Scatter Plot -- {title_suffix}")

    X_sample, labels_sample = _sample_data(X, labels, sample_size)

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=30, max_iter=500, learning_rate="auto")
    X_tsne = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    unique_labels = sorted(set(labels_sample))
    for label in unique_labels:
        mask = labels_sample == label
        color = "#999999" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                   alpha=0.5, s=8, edgecolors="none")

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(f"t-SNE Cluster Visualization -- {title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, loc="best")
    fig.tight_layout()

    filename = f"08_tsne_clusters_{title_suffix.lower().replace(' ', '_')}.png"
    _save_plot(fig, filename)


# =============================================================================
# Cluster Profile Visualizations
# =============================================================================
def plot_cluster_heatmap(cluster_profiles):
    """
    Plot a heatmap of mean feature values across clusters.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Mean feature values per cluster.
    """
    from sklearn.preprocessing import MinMaxScaler
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_profiles),
        columns=cluster_profiles.columns,
        index=[f"Cluster {i}" for i in cluster_profiles.index],
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(
        normalized, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Normalized Value"}, ax=ax,
    )
    ax.set_title("Cluster Feature Profiles (Normalized)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Cluster", fontsize=12)
    fig.tight_layout()
    _save_plot(fig, "09_cluster_heatmap.png")


def plot_cluster_radar(cluster_profiles):
    """
    Plot radar (spider) charts for each cluster showing feature profiles.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Mean feature values per cluster.
    """
    from sklearn.preprocessing import MinMaxScaler

    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_profiles),
        columns=cluster_profiles.columns,
        index=cluster_profiles.index,
    )

    features = list(normalized.columns)
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    n_clusters = len(normalized)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              subplot_kw=dict(polar=True))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, (cluster_id, row) in enumerate(normalized.iterrows()):
        ax = axes[idx]
        values = row.values.tolist()
        values += values[:1]

        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=8)
        ax.set_title(f"Cluster {cluster_id}",
                     fontsize=12, fontweight="bold", pad=20)
        ax.set_ylim(0, 1)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cluster Radar Profiles",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_plot(fig, "10_cluster_radar.png")


def plot_cluster_bar_comparison(cluster_profiles):
    """
    Plot grouped bar charts comparing mean feature values across clusters.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Mean feature values per cluster.
    """
    from sklearn.preprocessing import MinMaxScaler

    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_profiles),
        columns=cluster_profiles.columns,
        index=cluster_profiles.index,
    )

    n_clusters = len(normalized)
    n_features = len(normalized.columns)
    x = np.arange(n_features)
    width = 0.8 / n_clusters

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (cluster_id, row) in enumerate(normalized.iterrows()):
        offset = (i - n_clusters / 2 + 0.5) * width
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.bar(x + offset, row.values, width,
               label=f"Cluster {cluster_id}",
               color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Normalized Mean Value", fontsize=12)
    ax.set_title("Feature Comparison Across Clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(normalized.columns, rotation=45, ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_plot(fig, "11_cluster_bar_comparison.png")


def plot_feature_distributions_by_cluster(df, labels, features=None,
                                           sample_size=10000):
    """
    Plot per-cluster distribution (box plot) for each feature.

    Uses box plots instead of violin plots for speed on large datasets,
    and downsamples to sample_size rows.

    Parameters
    ----------
    df : pd.DataFrame
        Unscaled feature DataFrame.
    labels : array-like
        Cluster labels.
    features : list of str, optional
        Features to plot. Defaults to all columns.
    sample_size : int
        Max samples to use for the plot.
    """
    if features is None:
        features = list(df.columns)

    plot_df = df[features].copy()
    plot_df["Cluster"] = labels
    plot_df = plot_df[plot_df["Cluster"] != -1]

    # Downsample for rendering speed
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=RANDOM_STATE)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    n_unique = len(plot_df["Cluster"].unique())
    palette = CLUSTER_COLORS[:n_unique]

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(x="Cluster", y=feature, data=plot_df, ax=ax,
                    palette=palette, linewidth=0.8, fliersize=1)
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cluster")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Cluster",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_plot(fig, "12_feature_distributions_by_cluster.png")


def plot_cluster_sizes(labels):
    """
    Plot a bar chart showing the number of songs in each cluster.

    Parameters
    ----------
    labels : array-like
        Cluster labels.
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    colors = ["#999999" if u == -1 else CLUSTER_COLORS[u % len(CLUSTER_COLORS)]
              for u in unique]
    names = ["Noise" if u == -1 else f"Cluster {u}" for u in unique]

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50, f"{count:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Songs", fontsize=12)
    ax.set_title("Cluster Size Distribution",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "13_cluster_sizes.png")


# =============================================================================
# Hierarchical Clustering -- Dendrogram
# =============================================================================
def plot_dendrogram(X, method="ward", max_display=30, sample_size=1500):
    """
    Plot a dendrogram for hierarchical clustering.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    method : str
        Linkage method (ward, complete, average, single).
    max_display : int
        Maximum leaf nodes to show (truncation level).
    sample_size : int
        Sample size for large datasets.
    """
    print_subheader("Dendrogram")

    X_sample, _ = _sample_data(X, np.zeros(len(X)), sample_size)

    Z = linkage(X_sample, method=method)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    dendrogram(Z, truncate_mode="lastp", p=max_display, ax=ax,
               leaf_rotation=90, leaf_font_size=9,
               color_threshold=0.7 * max(Z[:, 2]))

    ax.set_xlabel("Sample Index (or cluster size)", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "14_dendrogram.png")


# =============================================================================
# Pre-computed variants (for use with cloud_compute.py results)
# =============================================================================
def plot_silhouette_diagram_precomputed(sil_values, sil_labels, n_clusters):
    """
    Plot silhouette diagram from PRE-COMPUTED per-sample silhouette values.

    Includes per-cluster average annotations on the right side, with
    amber highlighting for clusters below the overall average.

    Parameters
    ----------
    sil_values : np.ndarray
        Pre-computed silhouette coefficient for each sample.
    sil_labels : np.ndarray
        Cluster labels corresponding to sil_values.
    n_clusters : int
        Number of clusters.
    """
    print_subheader("Silhouette Diagram (pre-computed)")

    avg_score = float(np.mean(sil_values))

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    y_lower = 10
    for i in range(n_clusters):
        cluster_values = sil_values[sil_labels == i]
        cluster_values.sort()
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        cluster_avg = float(np.mean(cluster_values))

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.75)
        ax.text(-0.03, y_lower + 0.5 * size, str(i),
                fontsize=10, fontweight="bold")

        # Per-cluster annotation -- amber for weak clusters
        ann_color = "#D4760A" if cluster_avg < avg_score else "#333333"
        ax.text(max(cluster_values) + 0.02, y_lower + 0.5 * size,
                f"Cluster {i}: avg={cluster_avg:.3f}",
                fontsize=9, fontweight="bold", color=ann_color,
                verticalalignment="center")

        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="#FF6B6B", linestyle="--", linewidth=2,
               label=f"Avg Score: {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Silhouette Diagram -- Per-Sample Analysis\n"
                 "Cluster 0 shows weak silhouette "
                 "(many songs near cluster boundary)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_plot(fig, "06_silhouette_diagram.png")


def plot_tsne_precomputed(tsne_coords, labels, title_suffix="K-Means",
                          n_total=None):
    """
    Plot t-SNE scatter from PRE-COMPUTED 2D coordinates.

    Includes a subtitle showing the sample size when plotting a subsample.

    Parameters
    ----------
    tsne_coords : np.ndarray of shape (n_samples, 2)
        Pre-computed t-SNE 2D coordinates.
    labels : np.ndarray
        Cluster labels for each point.
    title_suffix : str
        Algorithm name for the title.
    n_total : int, optional
        Total dataset size (for the subtitle annotation).
    """
    print_subheader(f"t-SNE Scatter Plot (pre-computed) -- {title_suffix}")

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        color = "#999999" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], c=color,
                   label=name, alpha=0.4, s=8, edgecolors="none")

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

    n_shown = len(tsne_coords)
    subtitle = (f"\nVisualization on stratified sample of {n_shown:,} songs "
                "(preserves cluster ratios)")
    ax.set_title(f"t-SNE Cluster Visualization -- {title_suffix}{subtitle}",
                 fontsize=12, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, loc="best")
    fig.tight_layout()

    filename = f"08_tsne_clusters_{title_suffix.lower().replace(' ', '_')}.png"
    _save_plot(fig, filename)


def plot_dendrogram_precomputed(X, method="ward", max_display=30,
                                 sample_size=1000):
    """
    Plot a dendrogram using a small in-memory sample (fast, no heavy compute).

    Parameters
    ----------
    X : array-like
        Scaled feature matrix (will be sampled to sample_size).
    method : str
        Linkage method.
    max_display : int
        Max leaf nodes to show.
    sample_size : int
        Number of points to sample (default 1000 for speed).
    """
    print_subheader("Dendrogram (small sample)")

    X_sample, _ = _sample_data(X, np.zeros(len(X)), sample_size)

    Z = linkage(X_sample, method=method)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    dendrogram(Z, truncate_mode="lastp", p=max_display, ax=ax,
               leaf_rotation=90, leaf_font_size=9,
               color_threshold=0.7 * max(Z[:, 2]))

    ax.set_xlabel("Sample Index (or cluster size)", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "14_dendrogram.png")


# =============================================================================
# New Analysis Visualizations (Fixes 7, 8, 9)
# =============================================================================
def plot_genre_per_cluster(df, label_col="cluster_kmeans", labels_map=None):
    """
    Plot a stacked bar chart showing genre_family distribution per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'genre_family' and cluster label columns.
    label_col : str
        Column containing cluster labels.
    labels_map : dict, optional
        Mapping of cluster_id → descriptive label.
    """
    if 'genre_family' not in df.columns or label_col not in df.columns:
        print("  [SKIP] genre_family or cluster column not found.")
        return

    ct = pd.crosstab(df[label_col], df['genre_family'], normalize='index') * 100
    ct = ct[ct.mean().sort_values(ascending=False).index]  # sort by popularity

    fig, ax = plt.subplots(figsize=(14, 7))
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', edgecolor='white',
            linewidth=0.3)

    if labels_map:
        xlabels = [f"C{i}: {labels_map.get(i, i)}" for i in ct.index]
        ax.set_xticklabels(xlabels, rotation=30, ha='right')

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Genre Family Distribution per Cluster",
                 fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
              title="Genre Family")
    fig.tight_layout()
    _save_plot(fig, "15_genre_per_cluster.png")


def plot_genre_cluster_heatmap(df, label_col="cluster_kmeans", labels_map=None):
    """
    Plot a heatmap of genre_family % per cluster for validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'genre_family' and cluster label columns.
    label_col : str
        Column containing cluster labels.
    labels_map : dict, optional
        Mapping of cluster_id → descriptive label.
    """
    if 'genre_family' not in df.columns or label_col not in df.columns:
        print("  [SKIP] genre_family or cluster column not found.")
        return

    ct = pd.crosstab(df[label_col], df['genre_family'], normalize='index') * 100
    # Keep top 10 genre families for readability
    top_genres = ct.mean().sort_values(ascending=False).head(10).index
    ct = ct[top_genres]

    if labels_map:
        ct.index = [f"C{i}: {labels_map.get(i, i)}" for i in ct.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "% of Cluster"})
    ax.set_title("Genre Validation: % of Each Genre Family per Cluster",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Genre Family")
    fig.tight_layout()
    _save_plot(fig, "16_genre_cluster_heatmap.png")


def plot_popularity_by_cluster(df, label_col="cluster_kmeans", labels_map=None):
    """
    Plot a boxplot of popularity_songs distribution across clusters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'popularity_songs' and cluster label columns.
    label_col : str
        Column containing cluster labels.
    labels_map : dict, optional
        Mapping of cluster_id → descriptive label.

    Returns
    -------
    str
        One-line insight about the highest-popularity cluster.
    """
    if 'popularity_songs' not in df.columns or label_col not in df.columns:
        print("  [SKIP] popularity_songs or cluster column not found.")
        return ""

    plot_df = df[[label_col, 'popularity_songs']].copy()
    plot_df = plot_df[plot_df[label_col] != -1]

    if labels_map:
        plot_df['cluster_name'] = plot_df[label_col].map(
            lambda x: f"C{x}: {labels_map.get(x, x)}")
    else:
        plot_df['cluster_name'] = plot_df[label_col].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    order = sorted(plot_df['cluster_name'].unique())
    n_clusters = len(order)
    palette = CLUSTER_COLORS[:n_clusters]
    sns.boxplot(x='cluster_name', y='popularity_songs', data=plot_df,
                ax=ax, palette=palette, linewidth=0.8, fliersize=1)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Popularity Score", fontsize=12)
    ax.set_title("Song Popularity Distribution by Cluster",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    _save_plot(fig, "17_popularity_by_cluster.png")

    # Generate insight
    medians = plot_df.groupby(label_col)['popularity_songs'].median()
    best_c = medians.idxmax()
    best_label = labels_map.get(best_c, best_c) if labels_map else best_c
    insight = (f"Cluster {best_c} (\"{best_label}\") has the highest "
               f"median popularity at {medians[best_c]:.0f}.")
    print(f"  [INSIGHT] {insight}")
    return insight


def plot_decade_by_cluster(df, label_col="cluster_kmeans", labels_map=None):
    """
    Plot a grouped bar chart of release decade vs cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'decade' and cluster label columns.
    label_col : str
        Column containing cluster labels.
    labels_map : dict, optional
        Mapping of cluster_id → descriptive label.

    Returns
    -------
    str
        One-line insight about decade distribution.
    """
    if 'decade' not in df.columns or label_col not in df.columns:
        print("  [SKIP] decade or cluster column not found.")
        return ""

    plot_df = df[[label_col, 'decade']].dropna().copy()
    plot_df = plot_df[plot_df[label_col] != -1]
    plot_df['decade'] = plot_df['decade'].astype(int)

    # Filter to decades with enough data
    decade_counts = plot_df['decade'].value_counts()
    valid_decades = decade_counts[decade_counts >= 50].index
    plot_df = plot_df[plot_df['decade'].isin(valid_decades)]

    ct = pd.crosstab(plot_df['decade'], plot_df[label_col], normalize='index') * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    n_clusters = len(ct.columns)
    x = np.arange(len(ct.index))
    width = 0.8 / n_clusters

    for i, col in enumerate(ct.columns):
        offset = (i - n_clusters / 2 + 0.5) * width
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        lbl = f"C{col}: {labels_map.get(col, col)}" if labels_map else f"Cluster {col}"
        ax.bar(x + offset, ct[col].values, width, label=lbl,
               color=color, edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Decade", fontsize=12)
    ax.set_ylabel("% of Songs", fontsize=12)
    ax.set_title("Cluster Distribution Across Release Decades",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) + 's' for d in ct.index], rotation=45)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_plot(fig, "18_decade_by_cluster.png")

    # Generate insight
    dominant = ct.idxmax(axis=1)
    insight = f"Decade distribution shows temporal clustering patterns across {len(valid_decades)} decades."
    print(f"  [INSIGHT] {insight}")
    return insight

