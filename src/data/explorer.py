"""Interactive Data Explorer engine for InsightFlow.

Provides data transformation, multi-column search, categorical and numerical
filtering, sorting, and projection operations without mutating the source dataset.
"""

from typing import Any
import pandas as pd
import numpy as np


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify DataFrame columns into categorical, numerical, and datetime types.

    Returns:
        dict with keys 'numerical', 'categorical', and 'datetime'.
    """
    if df.empty or len(df.columns) == 0:
        return {"numerical": [], "categorical": [], "datetime": []}

    numerical_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols: list[str] = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Categorical includes object, category, bool, and string dtypes
    excluded = set(numerical_cols + datetime_cols)
    categorical_cols: list[str] = [str(col) for col in df.columns if col not in excluded]

    return {
        "numerical": [str(c) for c in numerical_cols],
        "categorical": categorical_cols,
        "datetime": [str(c) for c in datetime_cols],
    }


def search_dataset(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Perform a case-insensitive search across all columns.

    Args:
        df: Input DataFrame (will not be mutated).
        query: Search string.

    Returns:
        Filtered DataFrame containing rows that match the query in any column.
    """
    if df.empty or not query or not query.strip():
        return df

    cleaned_query = query.strip()
    mask = pd.Series(False, index=df.index)

    for col in df.columns:
        # Convert to string and perform literal substring match (regex=False)
        col_mask = (
            df[col]
            .astype(str)
            .str.contains(cleaned_query, case=False, na=False, regex=False)
        )
        mask = mask | col_mask

    return df[mask]


def filter_categorical(
    df: pd.DataFrame,
    column: str,
    selected_values: list[Any],
) -> pd.DataFrame:
    """Filter DataFrame by categorical column values.

    If selected_values is empty, all rows are returned (no filter applied).
    """
    if df.empty or column not in df.columns or not selected_values:
        return df

    # Match values either directly or converted to string for consistency
    return df[df[column].isin(selected_values)]


def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float,
    include_nulls: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame rows where numerical column falls within [min_val, max_val]."""
    if df.empty or column not in df.columns:
        return df

    col_series = pd.to_numeric(df[column], errors="coerce")
    range_mask = (col_series >= min_val) & (col_series <= max_val)

    if include_nulls:
        range_mask = range_mask | col_series.isna()

    return df[range_mask]


def sort_dataset(
    df: pd.DataFrame,
    column: str | None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Sort DataFrame by a specified column safely placing missing values last."""
    if df.empty or not column or column not in df.columns:
        return df

    return df.sort_values(by=column, ascending=ascending, na_position="last")


def select_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Project a subset of columns from the DataFrame.

    If columns is empty, returns original DataFrame.
    """
    if df.empty or not columns:
        return df

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return df

    return df[valid_cols]


def get_numeric_bounds(
    df: pd.DataFrame,
    column: str,
) -> tuple[float, float]:
    """Calculate the min and max values for a numerical column.

    Returns (min, max). If column has no valid numbers, returns (0.0, 1.0).
    """
    if df.empty or column not in df.columns:
        return 0.0, 1.0

    numeric_series = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric_series.empty:
        return 0.0, 1.0

    min_val = float(numeric_series.min())
    max_val = float(numeric_series.max())

    if min_val == max_val:
        # Provide small buffer so sliders don't collapse
        return min_val, min_val + 1.0

    return min_val, max_val


def get_unique_categories(
    df: pd.DataFrame,
    column: str,
    limit: int = 200,
) -> list[Any]:
    """Retrieve distinct non-null values for a categorical column, sorted."""
    if df.empty or column not in df.columns:
        return []

    unique_vals = df[column].dropna().unique()
    try:
        sorted_vals = sorted(unique_vals)
    except TypeError:
        # Fallback for mixed types
        sorted_vals = sorted(unique_vals, key=lambda x: str(x))

    return list(sorted_vals[:limit])
