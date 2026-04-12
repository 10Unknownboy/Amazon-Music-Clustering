# =============================================================================
# app.py — Streamlit Dashboard for Amazon Music Clustering
# =============================================================================
# Interactive web application that showcases clustering results, cluster
# profiles, evaluation metrics, and allows users to explore tracks by cluster.
# Run with:  streamlit run app.py
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import CLUSTERED_OUTPUT_PATH, CLUSTERING_FEATURES, PLOTS_DIR


# =============================================================================
# Page Configuration & Custom Styling
# =============================================================================
st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished, dark-themed look
st.markdown("""
<style>
    /* Main background & text */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    div[data-testid="stMetric"] label {
        color: #a0a0b0 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #4ECDC4 !important;
        font-weight: 700;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #e0e0e0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #a0a0b0;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(78, 205, 196, 0.15);
        color: #4ECDC4 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Selectbox & input styling */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* Info boxes */
    .info-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)


# Cluster color palette (matches visualization.py)
CLUSTER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]


# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_data():
    """Load the clustered output CSV, or return None if not found."""
    if os.path.exists(CLUSTERED_OUTPUT_PATH):
        return pd.read_csv(CLUSTERED_OUTPUT_PATH)
    return None


@st.cache_data
def get_cluster_profiles(df, features, label_col="cluster_kmeans"):
    """Compute mean feature values per cluster."""
    profile = df.groupby(label_col)[features].mean()
    return profile


@st.cache_data
def get_pca_data(df, features):
    """Compute PCA on the selected features."""
    X = df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar():
    """Render the sidebar navigation and return the selected page."""
    with st.sidebar:
        st.markdown("# 🎵 Music Clustering")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠 Overview", "📊 EDA", "🎯 Clustering", "📈 Profiles", "🔍 Explore"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<div style='color:#606070; font-size:12px; text-align:center;'>"
            "Amazon Music Clustering<br>Unsupervised ML Project"
            "</div>",
            unsafe_allow_html=True,
        )

    return page


# =============================================================================
# Pages
# =============================================================================
def page_overview(df):
    """Overview page with project summary and key metrics."""
    st.markdown("# 🎵 Amazon Music Clustering")
    st.markdown("##### Unsupervised ML-powered song grouping by audio features")
    st.markdown("---")

    if df is None:
        st.error(
            "⚠️ No clustered data found. Run `python main.py` first to generate results."
        )
        return

    # Key metrics row
    n_songs = len(df)
    n_clusters = df["cluster_kmeans"].nunique()
    n_features = len(CLUSTERING_FEATURES)
    n_artists = df["name_artists"].nunique() if "name_artists" in df.columns else "—"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Songs", f"{n_songs:,}")
    col2.metric("Clusters", n_clusters)
    col3.metric("Audio Features", n_features)
    col4.metric("Unique Artists", f"{n_artists:,}" if isinstance(n_artists, int) else n_artists)

    st.markdown("---")

    # Cluster distribution
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 📊 Cluster Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        cluster_counts = df["cluster_kmeans"].value_counts().sort_index()
        colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in cluster_counts.index]
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors,
                      edgecolor="white", linewidth=0.5)

        for bar, count in zip(bars, cluster_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{count:,}", ha="center", va="bottom", color="white",
                    fontsize=10, fontweight="bold")

        ax.set_xlabel("Cluster", color="white", fontsize=11)
        ax.set_ylabel("Songs", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("### 🏷️ Cluster Labels")
        if "cluster_label" in df.columns:
            label_df = df.groupby("cluster_kmeans")["cluster_label"].first().reset_index()
            label_df.columns = ["Cluster", "Description"]
            label_df["Count"] = [
                f"{cluster_counts.get(c, 0):,}" for c in label_df["Cluster"]
            ]
            st.dataframe(label_df, use_container_width=True, hide_index=True)

    # Features used
    st.markdown("---")
    st.markdown("### 🎛️ Clustering Features")
    feat_cols = st.columns(5)
    for i, feature in enumerate(CLUSTERING_FEATURES):
        feat_cols[i % 5].markdown(
            f"<div style='background:rgba(78,205,196,0.1); border:1px solid rgba(78,205,196,0.3); "
            f"border-radius:8px; padding:8px 12px; text-align:center; margin:4px 0;'>"
            f"<span style='color:#4ECDC4; font-weight:600;'>{feature}</span></div>",
            unsafe_allow_html=True,
        )


def page_eda(df):
    """Exploratory Data Analysis page."""
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    if df is None:
        st.error("⚠️ No data found. Run `python main.py` first.")
        return

    available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]

    # Feature statistics
    st.markdown("### 📋 Feature Statistics")
    stats_df = df[available_features].describe().round(3)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlations")
    corr = df[available_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="white")
    for text in ax.texts:
        text.set_color("white")
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature distributions
    st.markdown("### 📈 Feature Distributions")
    selected_feature = st.selectbox("Select a feature", available_features)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.hist(df[selected_feature], bins=60, color="#4ECDC4", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(selected_feature, color="white", fontsize=11)
    ax.set_ylabel("Count", color="white", fontsize=11)
    ax.set_title(f"Distribution of {selected_feature}", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()


def page_clustering(df):
    """Clustering results visualization page."""
    st.markdown("# 🎯 Clustering Results")
    st.markdown("---")

    if df is None:
        st.error("⚠️ No data found. Run `python main.py` first.")
        return

    available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]

    # Algorithm selector
    algo = st.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical"],
    )
    label_col_map = {
        "K-Means": "cluster_kmeans",
        "DBSCAN": "cluster_dbscan",
        "Hierarchical": "cluster_hierarchical",
    }
    label_col = label_col_map[algo]

    if label_col not in df.columns:
        st.warning(f"No labels found for {algo}. Run `python main.py` first.")
        return

    # PCA scatter plot
    st.markdown("### 🎨 PCA Cluster Visualization")

    X_pca, var_ratio = get_pca_data(df, available_features)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    labels = df[label_col].values
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        color = "#555555" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=0.4, s=6, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)", color="white", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)", color="white", fontsize=11)
    ax.set_title(f"{algo} — PCA Visualization", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(markerscale=4, fontsize=9, loc="best",
                       facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    st.pyplot(fig)
    plt.close()

    # Cluster sizes
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Cluster Sizes")
        size_df = df[label_col].value_counts().sort_index().reset_index()
        size_df.columns = ["Cluster", "Count"]
        size_df["Percentage"] = (size_df["Count"] / size_df["Count"].sum() * 100).round(1)
        st.dataframe(size_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 📐 Cluster Metrics")
        st.info("Run `python main.py` to compute full evaluation metrics (Silhouette, DBI, etc.)")

    # Saved plots gallery
    st.markdown("---")
    st.markdown("### 🖼️ Saved Visualizations")

    if os.path.exists(PLOTS_DIR):
        plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        if plot_files:
            selected_plot = st.selectbox("Select a plot", plot_files)
            plot_path = os.path.join(PLOTS_DIR, selected_plot)
            st.image(plot_path, use_container_width=True)
        else:
            st.info("No plots saved yet. Run `python main.py` to generate visualizations.")
    else:
        st.info("No plots directory found. Run `python main.py` first.")


def page_profiles(df):
    """Cluster profiling and interpretation page."""
    st.markdown("# 📈 Cluster Profiles")
    st.markdown("---")

    if df is None:
        st.error("⚠️ No data found. Run `python main.py` first.")
        return

    available_features = [f for f in CLUSTERING_FEATURES if f in df.columns]
    profiles = get_cluster_profiles(df, available_features)

    # Normalized heatmap
    st.markdown("### 🌡️ Cluster Feature Heatmap")
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(profiles),
        columns=profiles.columns,
        index=[f"Cluster {i}" for i in profiles.index],
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Normalized Feature Profiles", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Radar chart for a selected cluster
    st.markdown("### 🕸️ Cluster Radar Chart")
    cluster_ids = sorted(df["cluster_kmeans"].unique())
    selected_cluster = st.selectbox("Select cluster", cluster_ids)

    if selected_cluster in profiles.index:
        row = profiles.loc[selected_cluster]
        row_norm = (row - profiles.min()) / (profiles.max() - profiles.min())

        features_list = list(row_norm.index)
        values = row_norm.values.tolist()
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(features_list), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        color = CLUSTER_COLORS[selected_cluster % len(CLUSTER_COLORS)]
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features_list, fontsize=9, color="white")
        ax.set_ylim(0, 1)
        ax.tick_params(colors="white")
        ax.set_title(f"Cluster {selected_cluster}", color="white",
                     fontsize=14, fontweight="bold", pad=20)
        ax.grid(color="rgba(255,255,255,0.1)")
        st.pyplot(fig)
        plt.close()

    # Raw profile table
    st.markdown("---")
    st.markdown("### 📋 Raw Cluster Means")
    st.dataframe(profiles.round(3), use_container_width=True)


def page_explore(df):
    """Track exploration page — browse songs by cluster."""
    st.markdown("# 🔍 Explore Tracks")
    st.markdown("---")

    if df is None:
        st.error("⚠️ No data found. Run `python main.py` first.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        cluster_ids = sorted(df["cluster_kmeans"].unique())
        selected_cluster = st.selectbox("Filter by Cluster", ["All"] + cluster_ids)

    with col2:
        if "genres" in df.columns:
            all_genres = sorted(df["genres"].dropna().unique())
            selected_genre = st.selectbox("Filter by Genre", ["All"] + all_genres)
        else:
            selected_genre = "All"

    with col3:
        search_query = st.text_input("🔎 Search song or artist")

    # Apply filters
    filtered = df.copy()

    if selected_cluster != "All":
        filtered = filtered[filtered["cluster_kmeans"] == selected_cluster]

    if selected_genre != "All":
        filtered = filtered[filtered["genres"] == selected_genre]

    if search_query:
        query_lower = search_query.lower()
        mask = (
            filtered["name_song"].str.lower().str.contains(query_lower, na=False) |
            filtered["name_artists"].str.lower().str.contains(query_lower, na=False)
        )
        filtered = filtered[mask]

    # Display results
    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} tracks**")

    display_cols = ["name_song", "name_artists", "genres", "cluster_kmeans", "cluster_label"]
    display_cols += [f for f in CLUSTERING_FEATURES if f in filtered.columns]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].head(500),
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    st.markdown("---")
    csv_data = filtered[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_tracks.csv",
        mime="text/csv",
    )


# =============================================================================
# Main App Entry
# =============================================================================
def main():
    """Main Streamlit application entry point."""
    df = load_data()
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 EDA":
        page_eda(df)
    elif page == "🎯 Clustering":
        page_clustering(df)
    elif page == "📈 Profiles":
        page_profiles(df)
    elif page == "🔍 Explore":
        page_explore(df)


if __name__ == "__main__":
    main()
