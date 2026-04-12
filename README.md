# 🎵 Amazon Music Clustering

> **Unsupervised ML project** that clusters Amazon Music tracks using audio features. Applies EDA, preprocessing, K-Means, DBSCAN, Hierarchical Clustering, PCA & evaluation metrics to group songs into meaningful clusters for recommendations and playlist generation.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Use Cases](#-business-use-cases)
- [Domain](#-domain)
- [Skills & Takeaways](#-skills--takeaways)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Visualizations](#-key-visualizations)
- [Project Deliverables](#-project-deliverables)
- [Project Guidelines](#-project-guidelines)
- [License](#-license)

---

## 🎯 Problem Statement

With millions of songs available on platforms like Amazon, manually categorizing tracks into genres is impractical. The goal of this project is to **automatically group similar songs** based on their audio characteristics using clustering techniques. By analyzing patterns in features such as **tempo, energy, danceability**, and more, we develop a model that organizes songs into meaningful clusters — potentially representing different musical **genres or moods** — without any prior labels.

---

## 💼 Business Use Cases

| Use Case                        | Description                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Personalized Playlist Curation** | Automatically group songs that sound similar to enhance playlist generation.                     |
| **Improved Song Discovery**       | Suggest similar tracks to users based on their preferred audio profile.                          |
| **Artist Analysis**               | Help artists and producers identify competitive songs in the same audio cluster.                 |
| **Market Segmentation**           | Streaming platforms can analyze user listening patterns and optimize recommendations/promotions. |

---

## 🌐 Domain

**Music Analytics / Unsupervised Machine Learning**

---

## 🧠 Skills & Takeaways

- Data Exploration & Cleaning
- Feature Selection & Engineering
- Data Normalization (StandardScaler / MinMaxScaler)
- K-Means Clustering & Elbow Method
- DBSCAN & Hierarchical (Agglomerative) Clustering
- Silhouette Score & Davies-Bouldin Index
- PCA & t-SNE for Dimensionality Reduction
- Cluster Visualization & Interpretation
- Genre / Mood Inference from Clusters
- Data Storytelling & Reporting
- Python (Pandas, NumPy, scikit-learn, Matplotlib, Seaborn)

---

## 🛠️ Tech Stack

| Category           | Tools / Libraries                                    |
| ------------------ | ---------------------------------------------------- |
| **Language**       | Python 3.x                                           |
| **Data Handling**  | Pandas, NumPy                                        |
| **ML / Clustering**| scikit-learn (KMeans, DBSCAN, AgglomerativeClustering)|
| **Visualization**  | Matplotlib, Seaborn                                  |
| **Dimensionality** | PCA, t-SNE (sklearn.decomposition / sklearn.manifold) |
| **Evaluation**     | Silhouette Score, Davies-Bouldin Index, Inertia      |
| **Bonus App**      | Streamlit                                            |
| **Version Control**| Git / GitHub                                         |

---

## 📁 Project Structure

```
Amazon Music Clustering/
│
├── data/
│   ├── raw/                        # Original dataset(s)
│   │   └── single_genre_artists.csv
│   └── processed/                  # Cleaned & clustered output
│       └── clustered_songs.csv
│
├── src/
│   ├── data_preprocessing.py       # Loading, cleaning, scaling
│   ├── feature_selection.py        # Feature selection utilities
│   ├── clustering.py               # KMeans, DBSCAN, Hierarchical
│   ├── evaluation.py               # Silhouette, Davies-Bouldin, Inertia
│   ├── visualization.py            # All plotting functions
│   └── utils.py                    # Shared helper functions
│
├── outputs/
│   └── plots/                      # Saved visualization images
│
├── app.py                          # Streamlit dashboard (bonus)
├── main.py                         # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore rules
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+ installed on your system
- Git installed

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/10UNknownboy/Amazon-Music-Clustering.git
cd Amazon-Music-Clustering

# 2. Create & activate virtual environment
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Copy single_genre_artists.csv into data/raw/

# 5. Run the pipeline
python main.py

# 6. (Optional) Launch Streamlit dashboard
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

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
