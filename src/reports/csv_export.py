"""CSV export generator for InsightFlow.

Encodes DataFrames to UTF-8 CSV byte streams with safe handling of
empty datasets and special characters.
"""

import pandas as pd


def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame into UTF-8 encoded CSV bytes.

    Uses 'utf-8-sig' to ensure seamless encoding compatibility across
    Excel, text editors, and modern data tools.
    """
    if df is None:
        return b""

    # Handles empty DataFrames gracefully
    if df.empty and len(df.columns) == 0:
        return b""

    csv_string = df.to_csv(index=False)
    return csv_string.encode("utf-8-sig")
