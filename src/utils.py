# =============================================================================
# utils.py — Shared Configuration, Paths & Constants
# =============================================================================
# Centralizes all project paths, feature lists, and helper functions used
# across the clustering pipeline. Ensures directories exist on import.
# =============================================================================

import os


# ---------------------------------------------------------------------------
# Project directory paths
# ---------------------------------------------------------------------------
# Root of the project (two levels up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Output directories
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Dataset file paths
RAW_DATASET_PATH = os.path.join(RAW_DATA_DIR, "single_genre_artists.csv")
CLUSTERED_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "clustered_songs.csv")
CLUSTERED_OUTPUT_V2_PATH = os.path.join(PROCESSED_DATA_DIR, "clustered_songs_v2.csv")

# Cached results paths (renamed from cloud_* for clarity)
RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, "results.npz")
METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "metrics.json")
SUMMARY_REPORT_PATH = os.path.join(OUTPUTS_DIR, "cluster_summary_report.txt")


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# Audio features selected for clustering — these describe the rhythm, mood,
# instrumentation, and energy level of each track.
CLUSTERING_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Columns that serve as metadata / reference only — not used in clustering.
REFERENCE_COLUMNS = [
    "id_songs",
    "name_song",
    "id_artists",
    "name_artists",
    "genres",
    "release_date",
    "popularity_songs",
]

# Columns to drop entirely (not useful for clustering or metadata display)
DROP_COLUMNS = [
    "explicit",
    "key",
    "mode",
    "time_signature",
    "popularity_artists",
    "followers",
]


# ---------------------------------------------------------------------------
# Genre family mapping — collapses 3,153 raw genres into ~15 families
# ---------------------------------------------------------------------------
GENRE_MAP = {
    'jazz': 'Jazz', 'blues': 'Blues', 'pop': 'Pop',
    'rock': 'Rock', 'classical': 'Classical', 'opera': 'Classical',
    'hip hop': 'Hip-Hop', 'rap': 'Hip-Hop', 'r&b': 'R&B', 'soul': 'R&B',
    'electronic': 'Electronic', 'dance': 'Electronic', 'edm': 'Electronic',
    'country': 'Country', 'folk': 'Folk', 'indie': 'Indie',
    'hoerspiel': 'Spoken Word', 'podcast': 'Spoken Word',
    'metal': 'Metal', 'punk': 'Punk', 'latin': 'Latin',
    'reggae': 'Reggae', 'world': 'World Music',
}


def map_genre(raw):
    """Map a raw genre string to one of ~15 genre families via keyword matching."""
    raw = str(raw).lower()
    for key, family in GENRE_MAP.items():
        if key in raw:
            return family
    return 'Other'


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
# K-Means: range of cluster counts to search
KMEANS_K_RANGE = range(2, 11)

# DBSCAN defaults
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 10

# Hierarchical clustering defaults
HIERARCHICAL_N_CLUSTERS = 6
HIERARCHICAL_LINKAGE = "ward"

# Random seed for reproducibility
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Directory initialization — create output folders if they don't exist
# ---------------------------------------------------------------------------
def ensure_directories():
    """Create all required output directories if they do not exist."""
    for directory in [PROCESSED_DATA_DIR, PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)


# Run on import so directories are always available
ensure_directories()


# ---------------------------------------------------------------------------
# Console logging helpers
# ---------------------------------------------------------------------------
def print_header(title):
    """Print a formatted section header to the console."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    """Print a formatted sub-section header to the console."""
    print(f"\n--- {title} ---")


def print_metric(name, value, fmt=".4f"):
    """Print a single metric name-value pair."""
    if isinstance(value, float):
        print(f"  {name}: {value:{fmt}}")
    else:
        print(f"  {name}: {value}")