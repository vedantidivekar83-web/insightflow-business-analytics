"""Statistical analysis engine for InsightFlow.

Computes descriptive statistics, quartiles, and dispersion metrics for
numerical features without mutating the source dataset.
"""

from typing import Any
import pandas as pd
import numpy as np


def compute_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics for all numerical columns.

    Calculates:
        - Mean
        - Median
        - Mode (first mode)
        - Minimum and Maximum
        - Standard Deviation
        - Variance
        - Quartiles (Q1 25%, Q2 50%, Q3 75%, IQR)
        - Counts (Non-Null, Missing)

    Returns:
        DataFrame indexed by column name with statistical metrics as columns.
        Returns an empty DataFrame if no numerical columns exist or df is empty.
    """
    if df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) == 0:
        return pd.DataFrame()

    stats_records: list[dict[str, Any]] = []

    for col in num_df.columns:
        series = num_df[col].dropna()
        total_rows = len(df)
        non_null_count = len(series)
        missing_count = total_rows - non_null_count

        if non_null_count == 0:
            # Column exists but has zero non-null numeric values
            stats_records.append(
                {
                    "Column": str(col),
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Mode": np.nan,
                    "Std Dev": np.nan,
                    "Variance": np.nan,
                    "Min": np.nan,
                    "Q1 (25%)": np.nan,
                    "Median (Q2)": np.nan,
                    "Q3 (75%)": np.nan,
                    "Max": np.nan,
                    "IQR": np.nan,
                    "Non-Null Count": 0,
                    "Missing Count": missing_count,
                }
            )
            continue

        mean_val = float(series.mean())
        median_val = float(series.median())

        # Mode calculation (safe mode retrieval)
        mode_series = series.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else np.nan

        std_val = float(series.std(ddof=1)) if non_null_count > 1 else 0.0
        var_val = float(series.var(ddof=1)) if non_null_count > 1 else 0.0
        min_val = float(series.min())
        max_val = float(series.max())

        q1 = float(series.quantile(0.25))
        q2 = float(series.quantile(0.50))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        stats_records.append(
            {
                "Column": str(col),
                "Mean": round(mean_val, 4),
                "Median": round(median_val, 4),
                "Mode": round(mode_val, 4) if not np.isnan(mode_val) else np.nan,
                "Std Dev": round(std_val, 4),
                "Variance": round(var_val, 4),
                "Min": round(min_val, 4),
                "Q1 (25%)": round(q1, 4),
                "Median (Q2)": round(q2, 4),
                "Q3 (75%)": round(q3, 4),
                "Max": round(max_val, 4),
                "IQR": round(iqr, 4),
                "Non-Null Count": non_null_count,
                "Missing Count": missing_count,
            }
        )

    result_df = pd.DataFrame(stats_records)
    if not result_df.empty:
        result_df = result_df.set_index("Column")

    return result_df
