# Amazon Music Clustering

Unsupervised machine-learning pipeline that groups **95,837 songs** from the Amazon Music catalogue into meaningful clusters based on 10 audio features — without any genre labels.

---

## Problem Statement

With millions of songs available on streaming platforms, manually categorising tracks into genres is impractical. This project automatically groups similar songs using clustering techniques by analysing patterns in features such as tempo, energy, danceability, and more.

## Business Use Cases

| Use Case | Description |
|---|---|
| **Personalised Playlists** | Automatically curate playlists from same-cluster tracks |
| **Content Classification** | Replace or supplement manual genre tagging |
| **New Artist Discovery** | Recommend similar artists via cluster proximity |
| **Catalogue Analytics** | Understand catalogue composition across energy, mood, and era |

---

## Project Structure

```
Amazon Music Clustering/
├── data/
│   ├── raw/                        # Original dataset
│   │   └── single_genre_artists.csv
│   └── processed/                  # Pipeline outputs
│       ├── clustered_songs_v2.csv
│       ├── clustered_songs.csv
│       ├── results.npz
│       └── metrics.json
├── outputs/
│   ├── plots/                      # 20 generated visualisations
│   └── cluster_summary_report.txt
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Paths, constants, helpers
│   ├── data_preprocessing.py       # Load, clean, transform
│   ├── feature_selection.py        # Feature selection and correlation
│   ├── clustering.py               # K-Means, DBSCAN, Hierarchical
│   ├── evaluation.py               # Metrics and cluster profiling
│   └── visualization.py            # All plot functions (20 plots)
├── main.py                         # End-to-end pipeline
├── app.py                          # Streamlit dashboard (6 pages)
├── datainsights.py                 # Standalone EDA report generator
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** `single_genre_artists.csv`
- **Rows:** 95,837 songs
- **Columns:** 23 (10 used for clustering)

### Clustering Features

| Feature | Description |
|---|---|
| `danceability` | How suitable a track is for dancing (0-1) |
| `energy` | Perceptual measure of intensity and activity (0-1) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words (0-1) |
| `acousticness` | Likelihood the track is acoustic (0-1) |
| `instrumentalness` | Likelihood the track has no vocals (0-1) |
| `liveness` | Presence of a live audience (0-1) |
| `valence` | Musical positiveness / happiness (0-1) |
| `tempo` | Estimated tempo in BPM |
| `duration_ms` | Track duration (log-transformed) |

---

## Preprocessing Pipeline

1. **Log Transform** — `np.log1p(duration_ms)` to reduce heavy right skew
2. **Winsorisation** — Cap `speechiness` and `instrumentalness` at the 95th percentile
3. **Standard Scaling** — Zero-mean, unit-variance normalisation
4. **Genre Family Mapping** — 3,153 raw genres collapsed into 15 families for validation
5. **Decade Extraction** — Release year parsed into decade bins for temporal analysis

---

## Clustering Results

### Algorithm Comparison

| Algorithm | Silhouette | Davies-Bouldin | Clusters | Noise |
|---|---|---|---|---|
| **K-Means (k=3)** | 0.2431 | 1.5498 | 3 | - |
| DBSCAN | 0.2866 | 0.9564 | 2 | 3,428 |
| Hierarchical | 0.2454 | 1.5666 | 3 | - |

### Cluster Profiles (K-Means, k=3)

| Cluster | Label | Size | Key Traits |
|---|---|---|---|
| 0 | speech-heavy + live-feel | 12,787 (13.3%) | speechiness=0.81, liveness=0.43, 76.7% Spoken Word |
| 1 | high-energy | 52,177 (54.4%) | energy=0.69, loudness=-7.6, 36% Pop, 14% Rock |
| 2 | acoustic + low-energy | 30,873 (32.2%) | acousticness=0.75, valence=0.41, 31% Pop, 5% Folk |

### Key Findings

- Audio-only clustering independently recovered genre boundaries (Cluster 0 = 76.7% Spoken Word) without supervision
- Pre-1960s music dominated by Cluster 2 (acoustic); post-1970s by Cluster 1 (high-energy)
- k=3 selected over k=6 (silhouette 32.5% higher)

---

## Visualisations (20 Plots)

The pipeline generates 20 plots automatically saved to `outputs/plots/`:

| Plot | Description |
|---|---|
| 01 | Feature distributions (EDA) |
| 02 | Correlation heatmap |
| 03 | Feature boxplots |
| 04 | Elbow curve (inertia vs k) |
| 05 | Silhouette scores vs k |
| 06 | Silhouette diagram (per-sample, annotated) |
| 07 | PCA scatter (K-Means, DBSCAN, Hierarchical) |
| 08 | t-SNE scatter (10K stratified sample) |
| 09 | Cluster feature heatmap |
| 10 | Cluster radar charts |
| 11 | Cluster bar comparison |
| 12 | Feature distributions by cluster |
| 13 | Cluster sizes |
| 14 | Dendrogram |
| 15 | Genre family per cluster (stacked bar) |
| 16 | Genre-cluster heatmap |
| 17 | Popularity by cluster (boxplot) |
| 18 | Decade by cluster (grouped bar) |

---

## Streamlit Dashboard

Interactive web app with 6 pages:

| Page | Features |
|---|---|
| **Overview** | Key metrics, cluster distribution, auto-generated labels |
| **EDA** | Feature statistics, correlation heatmap, distributions |
| **Clustering** | PCA scatter for each algorithm, cluster sizes, saved plots |
| **Profiles** | Feature heatmap, radar chart, raw cluster means |
| **Explore** | Filter by cluster/genre/search, download CSV |
| **Recommender** | Song search, audio profile radar, top-10 similar songs |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/10UNknownboy/Amazon-Music-Clustering.git
cd Amazon-Music-Clustering
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Run K-Means, DBSCAN, and Hierarchical clustering
3. Generate 20 plots and a summary report
4. Export `clustered_songs_v2.csv`

### Launch the Dashboard

```bash
streamlit run app.py
```

---

## 📊 Dataset

| Property         | Details                                |
| ---------------- | -------------------------------------- |
| **File Name**    | `single_genre_artists.csv`             |
| **Location**     | `data/raw/`                            |
| **Description**  | Audio characteristics of Amazon Music songs — rhythm, mood, intensity & instrumentation. |

### Feature Descriptions

| Feature              | Type    | Description                                                    |
| -------------------- | ------- | -------------------------------------------------------------- |
| `track_id`           | Text    | Unique identifier for each track *(reference only)*            |
| `track_name`         | Text    | Name of the song *(reference only)*                            |
| `artist_name`        | Text    | Artist or band name *(reference only)*                         |
| `danceability`       | Float   | How suitable a track is for dancing (0.0 – 1.0)               |
| `energy`             | Float   | Intensity and activity measure (0.0 – 1.0)                    |
| `loudness`           | Float   | Overall loudness in decibels (dB)                              |
| `speechiness`        | Float   | Presence of spoken words (0.0 – 1.0)                          |
| `acousticness`       | Float   | Confidence that the track is acoustic (0.0 – 1.0)             |
| `instrumentalness`   | Float   | Predicts whether a track has no vocals (0.0 – 1.0)            |
| `liveness`           | Float   | Probability that the track was performed live (0.0 – 1.0)     |
| `valence`            | Float   | Musical positiveness / happiness (0.0 – 1.0)                  |
| `tempo`              | Float   | Estimated tempo in BPM                                         |
| `duration_ms`        | Integer | Duration of the track in milliseconds                          |

> **Note:** `track_id`, `track_name`, and `artist_name` are dropped before clustering — they serve as reference metadata only.

---

## 🔬 Approach

### Phase 1 — Data Exploration & Preprocessing
- Load dataset, inspect shape, dtypes, missing values, and duplicates
- Drop non-numeric reference columns (`track_id`, `track_name`, `artist_name`)
- Visualize feature distributions (histograms, box plots)
- Normalize features using `StandardScaler` (crucial for distance-based clustering)

### Phase 2 — Feature Selection
- Select audio-descriptive features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`
- Analyze feature correlations using heatmaps

### Phase 3 — Dimensionality Reduction *(Visualization Only)*
- **PCA** — Reduce to 2–3 components preserving maximum variance
- **t-SNE** — Capture complex non-linear relationships for scatter plots

### Phase 4 — Clustering

| Algorithm                 | Highlights                                                |
| ------------------------- | --------------------------------------------------------- |
| **K-Means**               | Simple, effective; optimal `k` via Elbow Method & Silhouette Score |
| **DBSCAN**                | Discovers arbitrary-shaped clusters; detects noise/outliers |
| **Hierarchical (Agglom.)**| Dendrogram visualization; no need to predefine `k`        |

### Phase 5 — Cluster Evaluation & Interpretation
- **Silhouette Score** — How close points are to their own cluster vs. others
- **Davies-Bouldin Index** — Lower = better separation
- **Inertia** — Cluster compactness (K-Means)
- Profile clusters by averaging features per cluster (e.g., *"Party tracks"*, *"Chill acoustic"*)

### Phase 6 — Visualization
- 2D scatter plots (PCA / t-SNE) with color-coded clusters
- Bar charts of average feature values per cluster
- Heatmaps comparing features across clusters
- Distribution plots per cluster

### Phase 7 — Final Analysis & Export
- Attach cluster labels to original DataFrame
- Show top tracks per cluster
- Export final CSV (`data/processed/clustered_songs.csv`)
- Summary report with cluster characteristics

---

## 📈 Evaluation Metrics

| Metric                     | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| **Silhouette Score**       | Measures how similar a song is to its own cluster vs. neighboring clusters. Higher is better (range: -1 to 1). |
| **Davies-Bouldin Index**   | Measures intra-cluster similarity and inter-cluster differences. Lower is better. |
| **Inertia (SSE)**          | Sum of squared distances to nearest cluster center. Used with Elbow Method. |
| **Cluster Size Balance**   | Evenness and distribution of songs across clusters.                      |
| **Feature Interpretability** | Clarity of dominant audio features per cluster.                         |

---

## 🎨 Key Visualizations

- **Elbow Curve** — SSE vs. number of clusters (k)
- **Silhouette Plot** — Per-sample silhouette widths
- **PCA Scatter Plot** — 2D cluster visualization
- **t-SNE Scatter Plot** — Non-linear 2D embedding
- **Cluster Feature Heatmap** — Mean feature values across clusters
- **Feature Distribution Box Plots** — Per-cluster distributions
- **Dendrogram** — Hierarchical clustering tree
- **Correlation Heatmap** — Feature inter-correlations

---

## 📦 Project Deliverables

| Deliverable                 | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| **Source Code** (`.py`)     | Modular scripts for preprocessing, clustering, visualization |
| **Final Report**            | Problem, approach, visualizations, cluster analysis      |
| **CSV Output**              | Final dataset with cluster labels (`clustered_songs.csv`)|
| **Saved Plots**             | All visualization outputs in `outputs/plots/`            |
| **Streamlit App** *(Bonus)* | Interactive dashboard showcasing clustering results      |

---

## 📝 Project Guidelines

- ✅ Use clean, modular code with well-defined functions
- ✅ Follow **PEP-8** style conventions
- ✅ Track changes using **Git / GitHub**
- ✅ Comment code blocks and explain each step clearly
- ✅ Save visual outputs (plots) and insights in the report
- ✅ Use a virtual environment (`env/`) for dependency isolation

---

## 🏷️ Technical Tags

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `KMeans` · `DBSCAN` · `Hierarchical Clustering` · `EDA` · `PCA` · `t-SNE` · `Unsupervised Learning` · `Recommendation` · `Streamlit`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.