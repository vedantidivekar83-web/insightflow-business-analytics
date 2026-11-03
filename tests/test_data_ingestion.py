"""Unit tests for InsightFlow data loading and validation engine.

Tests valid CSV, valid Excel, multi-sheet Excel, empty files,
invalid encodings, corrupt data, and size limits using standard unittest.
"""

import io
import unittest
import pandas as pd

from src.data.loader import (
    load_csv,
    load_excel,
    load_uploaded_dataset,
    inspect_excel_sheets,
)
from src.data.validator import (
    MAX_FILE_SIZE_BYTES,
    validate_file_extension,
    validate_file_size,
    validate_dataframe,
    extract_dataset_metadata,
)


class TestDataIngestion(unittest.TestCase):
    """Test suite covering Stage 4 data ingestion and validation."""

    def test_valid_csv_loading(self) -> None:
        """Verify that a standard CSV buffer is parsed correctly."""
        csv_data = "id,name,revenue,is_active\n1,Acme Corp,150000.50,True\n2,Beta LLC,89000.00,False\n"
        file_bytes = csv_data.encode("utf-8")

        df, err = load_csv(file_bytes)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "name", "revenue", "is_active"])

        # Test validation and metadata extraction
        valid, val_err = validate_dataframe(df)
        self.assertTrue(valid)
        self.assertIsNone(val_err)

        metadata = extract_dataset_metadata(df, "test.csv", len(file_bytes))
        self.assertEqual(metadata["total_rows"], 2)
        self.assertEqual(metadata["total_columns"], 4)

    def test_csv_encoding_fallback(self) -> None:
        """Verify that CSV with Latin-1 specific characters is handled properly."""
        csv_data = "id,city,metric\n1,München,100\n2,Zürich,200\n"
        file_bytes = csv_data.encode("latin1")

        df, err = load_csv(file_bytes)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("München", df["city"].values)

    def test_valid_excel_loading(self) -> None:
        """Verify that an Excel workbook with multiple sheets can be inspected and loaded."""
        buffer = io.BytesIO()
        df_sheet1 = pd.DataFrame({"product": ["Widget A", "Widget B"], "price": [19.99, 29.99]})
        df_sheet2 = pd.DataFrame({"region": ["North", "South"], "quota": [5000, 7500]})

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_sheet1.to_excel(writer, sheet_name="Products", index=False)
            df_sheet2.to_excel(writer, sheet_name="Regions", index=False)

        file_bytes = buffer.getvalue()

        # Inspect sheet names
        sheets = inspect_excel_sheets(file_bytes)
        self.assertEqual(sheets, ["Products", "Regions"])

        # Load first sheet
        df1, err1 = load_excel(file_bytes, sheet_name="Products")
        self.assertIsNone(err1)
        self.assertIsNotNone(df1)
        self.assertEqual(len(df1), 2)
        self.assertEqual(list(df1.columns), ["product", "price"])

        # Load second sheet
        df2, err2 = load_excel(file_bytes, sheet_name="Regions")
        self.assertIsNone(err2)
        self.assertIsNotNone(df2)
        self.assertEqual(len(df2), 2)
        self.assertEqual(list(df2.columns), ["region", "quota"])

    def test_empty_files(self) -> None:
        """Verify empty files (0 bytes and 0 rows) are detected gracefully."""
        # 0 bytes file validation
        val, err = validate_file_size(0)
        self.assertFalse(val)
        self.assertIn("empty", err.lower())

        # Empty CSV content
        df, err = load_csv(b"")
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

        # Header only CSV (0 data rows)
        df_header_only = pd.DataFrame(columns=["col1", "col2"])
        valid, val_err = validate_dataframe(df_header_only)
        self.assertFalse(valid)
        self.assertIn("zero data rows", val_err.lower())

    def test_invalid_file(self) -> None:
        """Verify corrupt/invalid bytes under supported extensions fail gracefully without crashing."""
        corrupt_bytes = b"\x00\x01\x02\x03This is not an excel file"

        # Parsing corrupt excel
        df, err = load_uploaded_dataset(corrupt_bytes, "corrupt.xlsx")
        self.assertIsNone(df)
        self.assertIsNotNone(err)
        self.assertIn("failed to read", err.lower())

        # Unsupported extension
        valid_ext, ext_err = validate_file_extension("data.pdf")
        self.assertFalse(valid_ext)
        self.assertIn("unsupported", ext_err.lower())

    def test_file_size_limit(self) -> None:
        """Verify that files over 50 MB are rejected."""
        valid_size, _ = validate_file_size(1024 * 1024)  # 1 MB
        self.assertTrue(valid_size)

        too_large_size = MAX_FILE_SIZE_BYTES + 1
        valid_large, large_err = validate_file_size(too_large_size)
        self.assertFalse(valid_large)
        self.assertIn("exceeds the maximum allowed limit", large_err.lower())


if __name__ == "__main__":
    unittest.main()
