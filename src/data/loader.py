"""Dataset loading and parsing engine for InsightFlow.

Supports in-memory ingestion of CSV, XLSX, and XLS datasets using Pandas.
"""

import io
from typing import Any
import pandas as pd


# Supported encodings to attempt when reading CSV datasets
CSV_ENCODINGS_TO_TRY: list[str] = [
    "utf-8",
    "utf-8-sig",
    "latin1",
    "cp1252",
    "iso-8859-1",
]


def inspect_excel_sheets(file_bytes: bytes) -> list[str]:
    """Inspect and return sheet names from an Excel binary stream.

    Returns an empty list if the workbook cannot be inspected.
    """
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        return list(excel_file.sheet_names)
    except Exception:
        return []


def load_csv(file_bytes: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV file buffer attempting multiple character encodings.

    Returns:
        tuple of (DataFrame, None) on success, or (None, error_message) on failure.
    """
    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS_TO_TRY:
        try:
            buffer = io.BytesIO(file_bytes)
            df = pd.read_csv(buffer, encoding=encoding)
            return df, None
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError:
            return None, "The uploaded CSV file is empty (contains no header or data)."
        except pd.errors.ParserError as exc:
            return None, f"CSV parsing error: {exc}"
        except Exception as exc:
            return None, f"Failed to read CSV dataset: {exc}"

    return None, f"Character encoding error: Unable to decode CSV with standard encodings. ({last_error})"


def load_excel(
    file_bytes: bytes,
    sheet_name: str | int | None = 0,
) -> tuple[pd.DataFrame | None, str | None]:
    """Read an Excel (.xlsx, .xls) workbook buffer.

    Returns:
        tuple of (DataFrame, None) on success, or (None, error_message) on failure.
    """
    try:
        buffer = io.BytesIO(file_bytes)
        df = pd.read_excel(buffer, sheet_name=sheet_name)
        if isinstance(df, dict):
            # If multi-sheet read returned dict of DataFrames, pick the first
            first_sheet = next(iter(df.values()))
            return first_sheet, None
        return df, None
    except pd.errors.EmptyDataError:
        return None, "The selected Excel sheet is empty."
    except Exception as exc:
        if "xlrd" in str(exc).lower():
            return (
                None,
                "Legacy Excel (.xls) files require the 'xlrd' library (not installed). "
                "Please convert the workbook to modern Excel (.xlsx) or export as CSV.",
            )
        return None, f"Failed to read Excel workbook: {exc}"


def load_uploaded_dataset(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | int | None = 0,
) -> tuple[pd.DataFrame | None, str | None]:
    """Orchestrate dataset loading based on file extension.

    Returns:
        tuple of (DataFrame, None) on success, or (None, error_message) on failure.
    """
    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        return load_csv(file_bytes)
    elif lower_name.endswith((".xlsx", ".xls")):
        return load_excel(file_bytes, sheet_name=sheet_name)
    else:
        return None, f"Unsupported file extension for file '{filename}'."
