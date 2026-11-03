"""Data quality assessment module for InsightFlow.

Audits missing cells, duplicate records, cardinality, and completeness percentages.
"""

from typing import Any
import pandas as pd


def compute_data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute dataset-level data quality metrics.

    Returns:
        dict containing total rows, total columns, missing cells, completeness,
        and duplicate record counts.
    """
    if df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "total_cells": 0,
            "missing_cells": 0,
            "missing_pct": 0.0,
            "completeness_pct": 100.0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
    completeness_pct = round(100.0 - missing_pct, 2)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "completeness_pct": completeness_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_pct, 2),
    }


def compute_column_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute column-level data quality breakdown.

    Columns:
        - Column: column name
        - Data Type: string representation of dtype
        - Total Rows: row count
        - Missing Count: null values
        - Missing (%): null percentage
        - Completeness (%): 100 - null percentage
        - Unique Values: distinct non-null count
        - Uniqueness (%): unique count / total rows

    Returns:
        DataFrame sorted by highest missing percentage.
    """
    if df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    records: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        completeness_pct = round(100.0 - missing_pct, 2)
        unique_count = int(series.nunique(dropna=True))
        uniqueness_pct = (unique_count / total_rows * 100) if total_rows > 0 else 0.0

        records.append(
            {
                "Column": str(col),
                "Data Type": str(series.dtype),
                "Total Rows": total_rows,
                "Missing Count": missing_count,
                "Missing (%)": round(missing_pct, 2),
                "Completeness (%)": completeness_pct,
                "Unique Values": unique_count,
                "Uniqueness (%)": round(uniqueness_pct, 2),
            }
        )

    quality_df = pd.DataFrame(records)
    if not quality_df.empty:
        quality_df = quality_df.sort_values(by="Missing (%)", ascending=False)

    return quality_df
