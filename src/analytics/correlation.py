"""Correlation analysis engine for InsightFlow.

Calculates Pearson correlation matrices, removes constant features,
handles insufficient data safely, and extracts top pairwise correlations.
"""

from typing import Any
import pandas as pd
import numpy as np


def compute_correlation_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute Pearson correlation matrix across non-constant numerical columns.

    Returns:
        tuple of (correlation_matrix_df, dropped_constant_columns).
        Returns an empty DataFrame if fewer than 2 numerical columns are suitable.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(), []

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        return pd.DataFrame(), []

    # Detect and drop constant columns (std == 0 or all NaN or single unique value)
    valid_cols: list[str] = []
    dropped_cols: list[str] = []

    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 2 or series.nunique() <= 1 or series.std() == 0.0:
            dropped_cols.append(str(col))
        else:
            valid_cols.append(str(col))

    if len(valid_cols) < 2:
        return pd.DataFrame(), dropped_cols

    corr_matrix = num_df[valid_cols].corr(method="pearson").round(4)
    corr_matrix = corr_matrix.fillna(0.0)
    for col in corr_matrix.columns:
        corr_matrix.loc[col, col] = 1.0

    return corr_matrix, dropped_cols


def get_top_correlations(
    corr_matrix: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Extract strongest pairwise correlations excluding diagonal self-correlations.

    Returns:
        List of dicts with keys 'feature_a', 'feature_b', 'correlation', and 'strength'.
    """
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return []

    pairs: list[dict[str, Any]] = []
    cols = list(corr_matrix.columns)
    seen: set[tuple[str, str]] = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            pair_key = (col_a, col_b)
            if pair_key not in seen:
                seen.add(pair_key)
                val = float(corr_matrix.loc[col_a, col_b])

                abs_val = abs(val)
                if abs_val >= 0.7:
                    strength = "Strong"
                elif abs_val >= 0.4:
                    strength = "Moderate"
                elif abs_val >= 0.2:
                    strength = "Weak"
                else:
                    strength = "Negligible"

                direction = "Positive" if val > 0 else "Negative"

                pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(val, 4),
                        "abs_correlation": round(abs_val, 4),
                        "relationship": f"{strength} {direction}",
                    }
                )

    # Sort by absolute correlation descending
    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return pairs[:top_n]
