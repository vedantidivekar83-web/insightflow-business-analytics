"""File and dataset validation utilities for InsightFlow.

Validates file extensions, size constraints, and DataFrame schema integrity.
"""

from typing import Any
import pandas as pd

# Upload constraints
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
SUPPORTED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls"}


def format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def validate_file_extension(filename: str) -> tuple[bool, str | None]:
    """Validate that the uploaded file has a supported extension."""
    if not filename:
        return False, "No file was provided."

    lower_name = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if lower_name.endswith(ext):
            return True, None

    supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return False, f"Unsupported file format. Please upload a file with one of these extensions: {supported_list}."


def validate_file_size(size_bytes: int) -> tuple[bool, str | None]:
    """Validate that the file size is within the allowable limit (50 MB)."""
    if size_bytes <= 0:
        return False, "The uploaded file is empty (0 bytes)."

    if size_bytes > MAX_FILE_SIZE_BYTES:
        max_str = format_file_size(MAX_FILE_SIZE_BYTES)
        actual_str = format_file_size(size_bytes)
        return False, f"File size ({actual_str}) exceeds the maximum allowed limit of {max_str}."

    return True, None


def validate_dataframe(df: pd.DataFrame | None) -> tuple[bool, str | None]:
    """Validate that the parsed DataFrame contains readable data."""
    if df is None:
        return False, "Unable to load dataset into memory."

    if not isinstance(df, pd.DataFrame):
        return False, "Parsed object is not a valid tabular DataFrame."

    if df.empty and len(df.columns) == 0:
        return False, "The uploaded dataset contains no columns or data records."

    if len(df.columns) == 0:
        return False, "The dataset has no identifiable columns."

    if len(df) == 0:
        return False, "The dataset contains column headers but zero data rows."

    return True, None


def extract_dataset_metadata(
    df: pd.DataFrame,
    filename: str,
    file_size_bytes: int,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Extract standard structural metadata from a validated DataFrame."""
    memory_bytes = int(df.memory_usage(deep=True).sum())

    schema_info = []
    for col in df.columns:
        non_null_count = int(df[col].notna().sum())
        total_rows = len(df)
        null_count = total_rows - non_null_count
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0

        schema_info.append(
            {
                "Column": str(col),
                "Data Type": str(df[col].dtype),
                "Non-Null Count": non_null_count,
                "Missing Values": null_count,
                "Completeness (%)": round(100.0 - null_pct, 1),
            }
        )

    return {
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "file_size_formatted": format_file_size(file_size_bytes),
        "sheet_name": sheet_name,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "memory_usage": format_file_size(memory_bytes),
        "schema_table": pd.DataFrame(schema_info),
    }
