# =============================================================================
# data_preprocessing.py — Data Loading, Cleaning & Scaling
# =============================================================================
# Handles the full preprocessing pipeline: loading the raw CSV, exploring its
# structure, handling missing values and duplicates, dropping non-clustering
# columns, and normalizing features using StandardScaler or MinMaxScaler.
# =============================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils import (
    RAW_DATASET_PATH,
    REFERENCE_COLUMNS,
    DROP_COLUMNS,
    print_header,
    print_subheader,
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
