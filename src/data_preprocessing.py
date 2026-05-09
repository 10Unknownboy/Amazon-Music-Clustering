# =============================================================================
# data_preprocessing.py — Data Loading, Cleaning & Scaling
# =============================================================================
# Handles the full preprocessing pipeline: loading the raw CSV, exploring its
# structure, handling missing values and duplicates, dropping non-clustering
# columns, and normalizing features using StandardScaler or MinMaxScaler.
# Includes log-transform for duration_ms and winsorization for outliers.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils import (
    RAW_DATASET_PATH,
    REFERENCE_COLUMNS,
    DROP_COLUMNS,
    print_header,
    print_subheader,
    map_genre,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(filepath=None):
    """
    Load the raw dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to RAW_DATASET_PATH from utils.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if filepath is None:
        filepath = RAW_DATASET_PATH

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {filepath}")
    print(f"       Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def explore_dataset(df):
    """
    Print a comprehensive summary of the dataset for initial exploration.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to explore.
    """
    print_header("DATA EXPLORATION")

    # Shape and columns
    print_subheader("Shape & Columns")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Column names: {list(df.columns)}")

    # Data types
    print_subheader("Data Types")
    for col in df.columns:
        print(f"  {col:25s} -> {df[col].dtype}")

    # Missing values
    print_subheader("Missing Values")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("  No missing values found [OK]")
    else:
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            print(f"  {col:25s} → {count:,} ({pct:.2f}%)")

    # Duplicates
    print_subheader("Duplicate Rows")
    dup_count = df.duplicated().sum()
    print(f"  Duplicate rows: {dup_count:,}")

    # Basic statistics for numeric columns
    print_subheader("Descriptive Statistics (numeric)")
    print(df.describe().round(3).to_string())


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def clean_dataset(df):
    """
    Clean the dataset by handling missing values and removing duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    initial_rows = len(df)

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_rows - len(df)

    # Drop rows with missing values in clustering-relevant columns
    df = df.dropna().reset_index(drop=True)
    dropped_na = initial_rows - dropped_dups - len(df)

    print(f"[INFO] Cleaning complete:")
    print(f"       Duplicates removed : {dropped_dups:,}")
    print(f"       NaN rows removed   : {dropped_na:,}")
    print(f"       Final shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


def drop_non_clustering_columns(df):
    """
    Separate the dataset into reference metadata and clustering-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - metadata: reference columns (track names, artist names, etc.)
        - features_df: numeric columns for clustering
    """
    # Preserve reference metadata for later use (track names, genres, etc.)
    available_ref_cols = [c for c in REFERENCE_COLUMNS if c in df.columns]
    metadata = df[available_ref_cols].copy()

    # Drop reference and non-useful columns
    cols_to_drop = [c for c in REFERENCE_COLUMNS + DROP_COLUMNS if c in df.columns]
    features_df = df.drop(columns=cols_to_drop)

    print(f"[INFO] Dropped {len(cols_to_drop)} non-clustering columns: {cols_to_drop}")
    print(f"       Remaining features: {list(features_df.columns)}")

    return metadata, features_df


# ---------------------------------------------------------------------------
# Feature transforms (Fix 1 & Fix 2)
# ---------------------------------------------------------------------------
def log_transform_duration(df):
    """
    Apply log1p transform to duration_ms to reduce extreme skewness.

    duration_ms has skewness ~10.0 and kurtosis ~239, making it problematic
    for distance-based clustering. log1p(x) = ln(1+x) compresses the tail.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'duration_ms' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with duration_ms log-transformed in place.
    """
    if 'duration_ms' in df.columns:
        original_max = df['duration_ms'].max()
        df['duration_ms'] = np.log1p(df['duration_ms'])
        print(f"[INFO] Log-transformed duration_ms: "
              f"max {original_max:,.0f} -> {df['duration_ms'].max():.2f}")
    return df


def winsorize_outliers(df, columns=None, upper_percentile=0.95):
    """
    Clip outlier-heavy features at the specified upper percentile.

    speechiness has 17.9% outliers and instrumentalness has 22.0% outliers
    (IQR method). Clipping at 95th percentile reduces their influence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to winsorize.
    columns : list of str, optional
        Columns to clip. Defaults to ['speechiness', 'instrumentalness'].
    upper_percentile : float
        The percentile at which to clip (default 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns clipped.
    """
    if columns is None:
        columns = ['speechiness', 'instrumentalness']

    for col in columns:
        if col in df.columns:
            threshold = df[col].quantile(upper_percentile)
            n_clipped = (df[col] > threshold).sum()
            df[col] = df[col].clip(upper=threshold)
            print(f"[INFO] Winsorized {col}: clipped {n_clipped:,} values "
                  f"at p{int(upper_percentile*100)} = {threshold:.4f}")
    return df


# ---------------------------------------------------------------------------
# Genre & decade enrichment (Fix 3 & Fix 9)
# ---------------------------------------------------------------------------
def add_genre_family(df):
    """
    Add a 'genre_family' column by mapping raw genres to ~15 families.

    Uses keyword string matching against the GENRE_MAP dictionary.
    This column is NOT used for clustering — only for post-hoc analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'genres' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'genre_family' column.
    """
    if 'genres' in df.columns:
        df['genre_family'] = df['genres'].apply(map_genre)
        n_families = df['genre_family'].nunique()
        print(f"[INFO] Genre family mapping: {n_families} families created")
        print(f"       Top families: "
              f"{dict(df['genre_family'].value_counts().head(5))}")
    return df


def add_decade(df):
    """
    Extract release decade from release_date and add as 'decade' column.

    Parses release_date to datetime, extracts year, and rounds down to decade.
    Used for post-hoc analysis (not clustering).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'release_date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'decade' column.
    """
    if 'release_date' in df.columns:
        dates = pd.to_datetime(df['release_date'], errors='coerce')
        df['decade'] = (dates.dt.year // 10 * 10).astype('Int64')
        valid = df['decade'].notna().sum()
        print(f"[INFO] Decade extraction: {valid:,} valid dates parsed, "
              f"decades: {sorted(df['decade'].dropna().unique().tolist())}")
    return df


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------
def scale_features(df, method="standard"):
    """
    Normalize features using StandardScaler or MinMaxScaler.

    Clustering algorithms are distance-based, so feature scaling is crucial
    to prevent features with larger ranges from dominating the distance
    calculations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing only numeric clustering features.
    method : str, optional
        Scaling method: 'standard' (z-score) or 'minmax' (0–1 range).
        Default is 'standard'.

    Returns
    -------
    tuple of (pd.DataFrame, scaler)
        - Scaled DataFrame with original column names preserved.
        - Fitted scaler object (for inverse transforms if needed).
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: '{method}'. Use 'standard' or 'minmax'.")

    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    print(f"[INFO] Features scaled using {method.upper()} scaler.")
    return scaled_df, scaler
