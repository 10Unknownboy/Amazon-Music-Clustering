# =============================================================================
# feature_selection.py — Feature Selection & Correlation Analysis
# =============================================================================
# Provides utilities to select the optimal subset of audio features for
# clustering and to compute correlation matrices for feature analysis.
# =============================================================================

import pandas as pd

from src.utils import CLUSTERING_FEATURES, print_subheader


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_clustering_features(df, features=None):
    """
    Select the subset of columns to be used for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all numeric features.
    features : list of str, optional
        List of column names to select. Defaults to CLUSTERING_FEATURES.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the selected clustering features.
    """
    if features is None:
        features = CLUSTERING_FEATURES

    # Only select features that actually exist in the DataFrame
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"[WARNING] Features not found in data: {missing}")

    selected_df = df[available].copy()
    print(f"[INFO] Selected {len(available)} clustering features: {available}")

    return selected_df


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------
def compute_correlation_matrix(df):
    """
    Compute the Pearson correlation matrix for the selected features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric features.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    corr_matrix = df.corr()

    print_subheader("Feature Correlation Highlights")

    # Find strong correlations (|r| > 0.5, excluding self-correlations)
    strong_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.5:
                strong_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], r)
                )

    if strong_pairs:
        for col1, col2, r in sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True):
            direction = "positive" if r > 0 else "negative"
            print(f"  {col1} <-> {col2}: r = {r:.3f} ({direction})")
    else:
        print("  No strong correlations (|r| > 0.5) found.")

    return corr_matrix


def get_feature_summary(df):
    """
    Generate a summary table of feature statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering features.

    Returns
    -------
    pd.DataFrame
        Summary with mean, std, min, max, and range for each feature.
    """
    summary = pd.DataFrame({
        "Mean": df.mean(),
        "Std": df.std(),
        "Min": df.min(),
        "Max": df.max(),
        "Range": df.max() - df.min(),
    })

    return summary.round(4)
