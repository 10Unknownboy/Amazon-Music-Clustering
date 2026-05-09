# =============================================================================
# app.py — Streamlit Dashboard for Amazon Music Clustering
# =============================================================================
# Interactive web app with 6 tabs including a Song Recommender (Fix 10).
# Run with:  streamlit run app.py
# =============================================================================

import os, sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import (CLUSTERED_OUTPUT_PATH, CLUSTERED_OUTPUT_V2_PATH,
                        CLUSTERING_FEATURES, PLOTS_DIR, RESULTS_PATH)

# =============================================================================
# Page Config & Styling
# =============================================================================
st.set_page_config(page_title="Amazon Music Clustering", page_icon="🎵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%); border-right: 1px solid rgba(255,255,255,0.05); }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; backdrop-filter: blur(10px); }
    div[data-testid="stMetric"] label { color: #a0a0b0 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #4ECDC4 !important; font-weight: 700; }
    h1 { background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
    h2, h3 { color: #e0e0e0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; color: #a0a0b0; }
    .stTabs [aria-selected="true"] { background: rgba(78, 205, 196, 0.15); color: #4ECDC4 !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_data():
    """Load the clustered output CSV (v2 preferred, fallback to v1)."""
    for path in [CLUSTERED_OUTPUT_V2_PATH, CLUSTERED_OUTPUT_PATH]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_data
def get_cluster_profiles(df, features, label_col="cluster_kmeans"):
    """Compute mean feature values per cluster."""
    return df.groupby(label_col)[features].mean()

@st.cache_data
def get_pca_data(df, features):
    """Compute PCA on the selected features."""
    X = df[features].values
    X_scaled = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca.explained_variance_ratio_

@st.cache_data
def load_scaled_features():
    """Load pre-computed scaled features from the results NPZ."""
    if os.path.exists(RESULTS_PATH):
        data = np.load(RESULTS_PATH, allow_pickle=True)
        if 'scaled_X' in data:
            return data['scaled_X']
    return None

# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.markdown("# 🎵 Music Clustering")
        st.markdown("---")
        page = st.radio("Navigate", ["🏠 Overview", "📊 EDA", "🎯 Clustering",
                                      "📈 Profiles", "🔍 Explore", "🎶 Recommender"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown("<div style='color:#606070; font-size:12px; text-align:center;'>"
                    "Amazon Music Clustering v2<br>Unsupervised ML</div>",
                    unsafe_allow_html=True)
    return page

# =============================================================================
# Pages
# =============================================================================
def page_overview(df):
    """Overview page with key metrics."""
    st.markdown("# 🎵 Amazon Music Clustering")
    st.markdown("##### Unsupervised ML-powered song grouping by audio features")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No clustered data found. Run `python main.py` first.")
        return

    n_songs = len(df)
    n_clusters = df["cluster_kmeans"].nunique()
    n_artists = df["name_artists"].nunique() if "name_artists" in df.columns else "—"
    n_genres = df["genre_family"].nunique() if "genre_family" in df.columns else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Songs", f"{n_songs:,}")
    c2.metric("Clusters", n_clusters)
    c3.metric("Unique Artists", f"{n_artists:,}" if isinstance(n_artists, int) else n_artists)
    c4.metric("Genre Families", n_genres)
    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("### 📊 Cluster Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
        cc = df["cluster_kmeans"].value_counts().sort_index()
        colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in cc.index]
        bars = ax.bar(cc.index, cc.values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, cc.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+50,
                    f"{count:,}", ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cluster", color="white"); ax.set_ylabel("Songs", color="white")
        ax.tick_params(colors="white")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["bottom","left"]: ax.spines[s].set_color("white")
        st.pyplot(fig); plt.close()

    with col_right:
        st.markdown("### 🏷️ Cluster Labels")
        if "cluster_label" in df.columns:
            label_df = df.groupby("cluster_kmeans")["cluster_label"].first().reset_index()
            label_df.columns = ["Cluster", "Description"]
            label_df["Count"] = [f"{cc.get(c, 0):,}" for c in label_df["Cluster"]]
            st.dataframe(label_df, width="stretch", hide_index=True)


def page_eda(df):
    """Exploratory Data Analysis page."""
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No data found."); return

    avail = [f for f in CLUSTERING_FEATURES if f in df.columns]
    st.markdown("### 📋 Feature Statistics")
    st.dataframe(df[avail].describe().round(3), width="stretch")
    st.markdown("---")

    st.markdown("### 🔗 Feature Correlations")
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.tick_params(colors="white")
    for t in ax.texts: t.set_color("white")
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=14, fontweight="bold")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 📈 Feature Distributions")
    sel = st.selectbox("Select a feature", avail)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    ax.hist(df[sel], bins=60, color="#4ECDC4", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(sel, color="white"); ax.set_ylabel("Count", color="white")
    ax.set_title(f"Distribution of {sel}", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_color("white")
    st.pyplot(fig); plt.close()


def page_clustering(df):
    """Clustering results visualization page."""
    st.markdown("# 🎯 Clustering Results")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No data found."); return

    avail = [f for f in CLUSTERING_FEATURES if f in df.columns]
    algo = st.selectbox("Select Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])
    col_map = {"K-Means": "cluster_kmeans", "DBSCAN": "cluster_dbscan",
               "Hierarchical": "cluster_hierarchical"}
    label_col = col_map.get(algo, "cluster_kmeans")
    if label_col not in df.columns:
        st.warning(f"No labels for {algo}."); return

    # Fix 7: DBSCAN info box
    if algo == "DBSCAN":
        st.info(
            "\u2139\ufe0f DBSCAN found 2 clusters with very uneven sizes. "
            "While DBSCAN scores better on silhouette (0.287) and Davies-Bouldin (0.956) "
            "than K-Means, it groups ~96% of songs into one cluster. "
            "Noise points (3,428 songs, ~3.6%) represent genre-ambiguous or outlier tracks. "
            "K-Means is preferred for practical use cases like playlist generation."
        )

    st.markdown("### 🎨 PCA Cluster Visualization")
    X_pca, var_ratio = get_pca_data(df, avail)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    labels = df[label_col].values
    for lbl in sorted(set(labels)):
        m = labels == lbl
        c = "#555555" if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
        n = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=c, label=n, alpha=0.4, s=6, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", color="white")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", color="white")
    ax.set_title(f"{algo} — PCA", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_color("white")
    ax.legend(markerscale=4, fontsize=9, loc="best", facecolor="#1a1a2e",
              edgecolor="white", labelcolor="white")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📊 Cluster Sizes")
        size_df = df[label_col].value_counts().sort_index().reset_index()
        size_df.columns = ["Cluster", "Count"]
        size_df["%"] = (size_df["Count"] / size_df["Count"].sum() * 100).round(1)
        st.dataframe(size_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### 🖼️ Saved Visualizations")
    if os.path.exists(PLOTS_DIR):
        plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        if plots:
            sel = st.selectbox("Select a plot", plots)
            st.image(os.path.join(PLOTS_DIR, sel), width="stretch")


def page_profiles(df):
    """Cluster profiling page."""
    st.markdown("# 📈 Cluster Profiles")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No data found."); return

    avail = [f for f in CLUSTERING_FEATURES if f in df.columns]
    profiles = get_cluster_profiles(df, avail)

    st.markdown("### 🌡️ Cluster Feature Heatmap")
    norm = pd.DataFrame(MinMaxScaler().fit_transform(profiles),
                        columns=profiles.columns,
                        index=[f"Cluster {i}" for i in profiles.index])
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Normalized Feature Profiles", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 🕸️ Cluster Radar Chart")
    cluster_ids = sorted(df["cluster_kmeans"].unique())
    sel_c = st.selectbox("Select cluster", cluster_ids)
    if sel_c in profiles.index:
        row = profiles.loc[sel_c]
        row_n = (row - profiles.min()) / (profiles.max() - profiles.min())
        feats = list(row_n.index); vals = row_n.values.tolist(); vals += vals[:1]
        angs = np.linspace(0, 2*np.pi, len(feats), endpoint=False).tolist(); angs += angs[:1]
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
        color = CLUSTER_COLORS[sel_c % len(CLUSTER_COLORS)]
        ax.fill(angs, vals, color=color, alpha=0.25)
        ax.plot(angs, vals, color=color, linewidth=2.5)
        ax.set_xticks(angs[:-1]); ax.set_xticklabels(feats, fontsize=7, color="white")
        ax.set_ylim(0, 1); ax.tick_params(axis='y', labelsize=6, colors="white")
        ax.set_title(f"Cluster {sel_c}", color="white", fontsize=12, fontweight="bold", pad=15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close()

    st.markdown("---")
    st.markdown("### 📋 Raw Cluster Means")
    st.dataframe(profiles.round(3), width="stretch")


def page_explore(df):
    """Track exploration page."""
    st.markdown("# 🔍 Explore Tracks")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No data found."); return

    c1, c2, c3 = st.columns(3)
    with c1:
        cids = sorted(df["cluster_kmeans"].unique())
        sel_c = st.selectbox("Filter by Cluster", ["All"] + cids)
    with c2:
        if "genre_family" in df.columns:
            gfs = sorted(df["genre_family"].dropna().unique())
            sel_g = st.selectbox("Filter by Genre Family", ["All"] + gfs)
        else:
            sel_g = "All"
    with c3:
        q = st.text_input("🔎 Search song or artist")

    filtered = df.copy()
    if sel_c != "All":
        filtered = filtered[filtered["cluster_kmeans"] == sel_c]
    if sel_g != "All" and "genre_family" in filtered.columns:
        filtered = filtered[filtered["genre_family"] == sel_g]
    if q:
        ql = q.lower()
        mask = (filtered["name_song"].str.lower().str.contains(ql, na=False) |
                filtered["name_artists"].str.lower().str.contains(ql, na=False))
        filtered = filtered[mask]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} tracks**")
    display_cols = ["name_song", "name_artists", "cluster_kmeans", "cluster_label"]
    if "genre_family" in filtered.columns:
        display_cols.insert(2, "genre_family")
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(500), width="stretch", hide_index=True)

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("📥 Download Filtered (CSV)", csv, "filtered_tracks.csv", "text/csv")


def page_recommender(df):
    """Song Recommender tab (Fix 10) — find similar songs by Euclidean distance."""
    st.markdown("# 🎶 Song Recommender")
    st.markdown("##### Find similar songs based on audio features")
    st.markdown("---")
    if df is None:
        st.error("⚠️ No data found. Run `python main.py` first.")
        return

    avail = [f for f in CLUSTERING_FEATURES if f in df.columns]
    if not avail:
        st.error("No clustering features found."); return

    # Load pre-computed scaled features or scale on the fly
    scaled_X = load_scaled_features()
    if scaled_X is None or len(scaled_X) != len(df):
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(df[avail].values)

    query = st.text_input("🔎 Type a song name (partial match)")
    if not query:
        st.info("Type a song name above to find similar tracks.")
        return

    matches = df[df["name_song"].str.lower().str.contains(query.lower(), na=False)]
    if matches.empty:
        st.warning(f"No songs found matching '{query}'.")
        return

    st.markdown(f"**Found {len(matches):,} matches. Select one:**")
    options = matches[["name_song", "name_artists"]].head(20)
    options["display"] = options["name_song"] + " — " + options["name_artists"]
    sel_idx = st.selectbox("Select song", options.index,
                           format_func=lambda i: options.loc[i, "display"])

    song_row = df.loc[sel_idx]
    song_cluster = song_row.get("cluster_kmeans", None)
    song_label = song_row.get("cluster_label", "Unknown")

    # Display selected song info
    st.markdown("---")
    st.markdown(f"### 🎵 {song_row['name_song']}")
    st.markdown(f"**Artist:** {song_row.get('name_artists', 'Unknown')}  |  "
                f"**Cluster:** {song_cluster} ({song_label})")
    if "genre_family" in song_row:
        st.markdown(f"**Genre Family:** {song_row.get('genre_family', 'Unknown')}")

    # Radar chart of the selected song
    song_vals = song_row[avail].values.astype(float)
    profiles = get_cluster_profiles(df, avail)
    song_norm = (song_vals - df[avail].min().values) / (df[avail].max().values - df[avail].min().values + 1e-10)
    feats = list(avail); vals = song_norm.tolist(); vals += vals[:1]
    angs = np.linspace(0, 2*np.pi, len(feats), endpoint=False).tolist(); angs += angs[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    ax.fill(angs, vals, color="#4ECDC4", alpha=0.25)
    ax.plot(angs, vals, color="#4ECDC4", linewidth=2.5)
    ax.set_xticks(angs[:-1]); ax.set_xticklabels(feats, fontsize=7, color="white")
    ax.set_ylim(0, 1); ax.tick_params(axis='y', labelsize=6, colors="white")
    ax.set_title("Audio Profile", color="white", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False); plt.close()

    # Find nearest neighbors within the same cluster
    st.markdown("---")
    st.markdown("### 🔗 Top 10 Similar Songs")

    if song_cluster is not None:
        cluster_mask = df["cluster_kmeans"] == song_cluster
        cluster_indices = df.index[cluster_mask].tolist()
    else:
        cluster_indices = df.index.tolist()

    # Remove the selected song from candidates
    cluster_indices = [i for i in cluster_indices if i != sel_idx]
    if not cluster_indices:
        st.info("No other songs in this cluster."); return

    # Compute Euclidean distances in scaled feature space
    song_vec = scaled_X[sel_idx].reshape(1, -1)
    candidate_vecs = scaled_X[cluster_indices]
    dists = euclidean_distances(song_vec, candidate_vecs).ravel()

    # Get top 10 nearest
    top_k = min(10, len(dists))
    top_indices = np.argsort(dists)[:top_k]
    top_df_indices = [cluster_indices[i] for i in top_indices]
    top_dists = dists[top_indices]

    # Build results dataframe
    result_cols = ["name_song", "name_artists"]
    if "genre_family" in df.columns:
        result_cols.append("genre_family")
    if "cluster_label" in df.columns:
        result_cols.append("cluster_label")
    result_cols = [c for c in result_cols if c in df.columns]

    result_df = df.loc[top_df_indices, result_cols].copy()
    result_df["similarity_score"] = (1 / (1 + top_dists)).round(4)
    result_df = result_df.reset_index(drop=True)
    result_df.index = range(1, len(result_df) + 1)
    result_df.index.name = "Rank"

    st.dataframe(result_df, width="stretch")


# =============================================================================
# Main
# =============================================================================
def main():
    """Main Streamlit application entry point."""
    df = load_data()
    page = render_sidebar()

    if page == "🏠 Overview": page_overview(df)
    elif page == "📊 EDA": page_eda(df)
    elif page == "🎯 Clustering": page_clustering(df)
    elif page == "📈 Profiles": page_profiles(df)
    elif page == "🔍 Explore": page_explore(df)
    elif page == "🎶 Recommender": page_recommender(df)

if __name__ == "__main__":
    main()
