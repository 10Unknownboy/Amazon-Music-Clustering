# =============================================================================
# visualization.py — Cluster Visualization & Plotting
# =============================================================================
# All plotting functions for the clustering pipeline. Generates publication-
# quality visualizations for EDA, cluster analysis, and evaluation.
# Every plot is saved to outputs/plots/ and optionally displayed.
# =============================================================================

import os
import numpy as np
import pandas as pd
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
# Use a clean, modern aesthetic for all plots
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


# =============================================================================
# EDA Visualizations
# =============================================================================
def plot_feature_distributions_eda(df, features=None):
    """
    Plot histograms and KDE curves for each audio feature.

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
        ax.hist(df[feature], bins=50, alpha=0.7, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                edgecolor="white", linewidth=0.5)
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (Raw Data)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_plot(fig, "01_feature_distributions.png")


def plot_correlation_heatmap(corr_matrix):
    """
    Plot a triangular correlation heatmap of audio features.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix from feature_selection.compute_correlation_matrix().
    """
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save_plot(fig, "02_correlation_heatmap.png")


def plot_boxplots(df, features=None):
    """
    Plot box plots for each feature to show spread and outliers.

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

    # Need to normalize for comparable box plots
    from sklearn.preprocessing import MinMaxScaler
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[features]),
        columns=features
    )

    scaled.boxplot(ax=ax, grid=False, patch_artist=True,
                   boxprops=dict(facecolor="#4ECDC4", alpha=0.7),
                   medianprops=dict(color="#FF6B6B", linewidth=2))
    ax.set_title("Feature Distributions (Normalized for Comparison)", fontsize=14, fontweight="bold")
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

    ax.plot(k_range, inertias, "o-", color="#45B7D1", linewidth=2.5, markersize=8,
            markerfacecolor="#FF6B6B", markeredgecolor="white", markeredgewidth=2)

    if optimal_k is not None:
        idx = list(k_range).index(optimal_k)
        ax.axvline(x=optimal_k, color="#FF6B6B", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.scatter([optimal_k], [inertias[idx]], s=200, color="#FF6B6B",
                   zorder=5, edgecolors="white", linewidths=2)
        ax.annotate(f"Optimal k = {optimal_k}", xy=(optimal_k, inertias[idx]),
                    xytext=(optimal_k + 0.5, inertias[idx]),
                    fontsize=12, fontweight="bold", color="#FF6B6B")

    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (SSE)", fontsize=12)
    ax.set_title("Elbow Method — Optimal Number of Clusters", fontsize=14, fontweight="bold")
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

    colors = ["#FF6B6B" if (optimal_k and k == optimal_k) else "#4ECDC4" for k in k_range]
    bars = ax.bar(k_range, scores, color=colors, edgecolor="white", linewidth=1.5, alpha=0.85)

    # Add value labels on top of bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs. Number of Clusters", fontsize=14, fontweight="bold")
    ax.set_xticks(k_range)
    fig.tight_layout()
    _save_plot(fig, "05_silhouette_scores.png")


def plot_silhouette_diagram(X, labels):
    """
    Plot a per-sample silhouette analysis diagram.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print("  [SKIP] Silhouette diagram requires at least 2 clusters.")
        return

    sample_silhouette_values = silhouette_samples(X, labels)
    avg_score = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    y_lower = 10
    for i in range(n_clusters):
        cluster_values = sample_silhouette_values[labels == i]
        cluster_values.sort()
        size = cluster_values.shape[0]
        y_upper = y_lower + size

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.75)
        ax.text(-0.03, y_lower + 0.5 * size, str(i), fontsize=10, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="#FF6B6B", linestyle="--", linewidth=2,
               label=f"Avg Score: {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Silhouette Diagram — Per-Sample Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_plot(fig, "06_silhouette_diagram.png")


# =============================================================================
# Cluster Visualizations (PCA & t-SNE)
# =============================================================================
def plot_pca_clusters(X, labels, title_suffix="K-Means"):
    """
    Reduce features to 2D using PCA and plot color-coded clusters.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    title_suffix : str
        Label to add to the plot title (e.g., algorithm name).
    """
    print_subheader(f"PCA Scatter Plot — {title_suffix}")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        color = "#999999" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=0.5, s=8, edgecolors="none")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=12)
    ax.set_title(f"PCA Cluster Visualization — {title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, loc="best")
    fig.tight_layout()

    filename = f"07_pca_clusters_{title_suffix.lower().replace(' ', '_')}.png"
    _save_plot(fig, filename)

    return pca


def plot_tsne_clusters(X, labels, title_suffix="K-Means", sample_size=10000):
    """
    Reduce features to 2D using t-SNE and plot color-coded clusters.

    t-SNE is computationally expensive, so we sample the data if too large.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    labels : array-like
        Cluster labels.
    title_suffix : str
        Algorithm name for the title.
    sample_size : int
        Max number of points to plot (t-SNE is slow on large datasets).
    """
    print_subheader(f"t-SNE Scatter Plot — {title_suffix}")

    # Sample if dataset is too large for t-SNE performance
    if len(X) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        indices = rng.choice(len(X), sample_size, replace=False)
        X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices].values
        labels_sample = np.array(labels)[indices]
        print(f"  Sampled {sample_size:,} / {len(X):,} points for t-SNE.")
    else:
        X_sample = X if isinstance(X, np.ndarray) else X.values
        labels_sample = np.array(labels)

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
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
    ax.set_title(f"t-SNE Cluster Visualization — {title_suffix}", fontsize=14, fontweight="bold")
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
        Mean feature values per cluster (from evaluation.profile_clusters()).
    """
    # Normalize profiles to 0-1 range for fair color comparison
    from sklearn.preprocessing import MinMaxScaler
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_profiles),
        columns=cluster_profiles.columns,
        index=[f"Cluster {i}" for i in cluster_profiles.index],
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Normalized Value"},
        ax=ax,
    )
    ax.set_title("Cluster Feature Profiles (Normalized)", fontsize=14, fontweight="bold", pad=15)
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

    # Normalize for radar plot
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_profiles),
        columns=cluster_profiles.columns,
        index=cluster_profiles.index,
    )

    features = list(normalized.columns)
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    n_clusters = len(normalized)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
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
        ax.set_title(f"Cluster {cluster_id}", fontsize=12, fontweight="bold", pad=20)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cluster Radar Profiles", fontsize=16, fontweight="bold", y=1.02)
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
        ax.bar(x + offset, row.values, width, label=f"Cluster {cluster_id}",
               color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Normalized Mean Value", fontsize=12)
    ax.set_title("Feature Comparison Across Clusters", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(normalized.columns, rotation=45, ha="right")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_plot(fig, "11_cluster_bar_comparison.png")


def plot_feature_distributions_by_cluster(df, labels, features=None):
    """
    Plot per-cluster distribution (violin plot) for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Unscaled feature DataFrame.
    labels : array-like
        Cluster labels.
    features : list of str, optional
        Features to plot. Defaults to all columns.
    """
    if features is None:
        features = list(df.columns)

    plot_df = df[features].copy()
    plot_df["Cluster"] = labels
    plot_df = plot_df[plot_df["Cluster"] != -1]  # Exclude noise

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.violinplot(x="Cluster", y=feature, data=plot_df, ax=ax,
                       palette=CLUSTER_COLORS[:len(plot_df["Cluster"].unique())],
                       inner="box", linewidth=0.8)
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cluster")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Cluster", fontsize=16, fontweight="bold", y=1.02)
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

    colors = ["#999999" if u == -1 else CLUSTER_COLORS[u % len(CLUSTER_COLORS)] for u in unique]
    names = ["Noise" if u == -1 else f"Cluster {u}" for u in unique]

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5)

    # Add count labels on top
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Songs", fontsize=12)
    ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "13_cluster_sizes.png")


# =============================================================================
# Hierarchical Clustering — Dendrogram
# =============================================================================
def plot_dendrogram(X, method="ward", max_display=30, sample_size=2000):
    """
    Plot a dendrogram for hierarchical clustering.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    method : str
        Linkage method (ward, complete, average, single).
    max_display : int
        Maximum number of leaf nodes to show (truncation level).
    sample_size : int
        Sample size for large datasets (dendrograms are slow on large data).
    """
    print_subheader("Dendrogram")

    # Sample if too large
    if len(X) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        indices = rng.choice(len(X), sample_size, replace=False)
        X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices].values
        print(f"  Sampled {sample_size:,} / {len(X):,} points for dendrogram.")
    else:
        X_sample = X if isinstance(X, np.ndarray) else X.values

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
