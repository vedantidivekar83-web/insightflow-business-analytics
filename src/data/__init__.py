"""Data ingestion, validation, and preprocessing modules.

Responsible for dataset loading (CSV/Excel), schema inspection, and data cleaning.
"""

from src.data.loader import (
    load_csv,
    load_excel,
    load_uploaded_dataset,
    inspect_excel_sheets,
)
from src.data.validator import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    format_file_size,
    validate_file_extension,
    validate_file_size,
    validate_dataframe,
    extract_dataset_metadata,
)
from src.data.explorer import (
    classify_columns,
    search_dataset,
    filter_categorical,
    filter_numeric_range,
    sort_dataset,
    select_columns,
    get_numeric_bounds,
    get_unique_categories,
)

__all__ = [
    "load_csv",
    "load_excel",
    "load_uploaded_dataset",
    "inspect_excel_sheets",
    "MAX_FILE_SIZE_BYTES",
    "SUPPORTED_EXTENSIONS",
    "format_file_size",
    "validate_file_extension",
    "validate_file_size",
    "validate_dataframe",
    "extract_dataset_metadata",
    "classify_columns",
    "search_dataset",
    "filter_categorical",
    "filter_numeric_range",
    "sort_dataset",
    "select_columns",
    "get_numeric_bounds",
    "get_unique_categories",
]
