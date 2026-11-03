"""Unit test suite for Stage 8: Reports & Export Module.

Tests CSV byte generation, multi-sheet Excel workbook creation (Openpyxl),
empty datasets, special characters, long text handling, and PDF limitations.
"""

import io
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import openpyxl

from src.reports.csv_export import generate_csv_bytes
from src.reports.excel_export import generate_excel_workbook_bytes
from src.reports.pdf_export import is_pdf_export_available, get_pdf_limitation_message
from src.ui.views.reports import render_reports_view


class TestReportsModule(unittest.TestCase):
    """Test suite covering data export and report generation."""

    def setUp(self) -> None:
        """Create sample dataset for testing exports."""
        self.sample_df = pd.DataFrame(
            {
                "transaction_id": [101, 102, 103],
                "product_name": ["Widget A & Co.", "Gadget (Special) #1", "Item <Test>"],
                "quantity": [2, 5, 1],
                "unit_price": [19.99, 29.50, 100.00],
                "status": ["Delivered", "Delivered", "Cancelled"],
            }
        )

    def test_csv_export_content_and_encoding(self) -> None:
        """Verify CSV bytes are properly UTF-8 encoded with correct headers and rows."""
        csv_bytes = generate_csv_bytes(self.sample_df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertTrue(len(csv_bytes) > 0)

        # Decode and parse back
        decoded_df = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(decoded_df), 3)
        self.assertIn("product_name", decoded_df.columns)
        self.assertEqual(decoded_df.iloc[0]["product_name"], "Widget A & Co.")

    def test_csv_special_characters(self) -> None:
        """Verify CSV export preserves non-ASCII and special characters."""
        special_df = pd.DataFrame(
            {
                "col_ümlaut": ["München", "Zürich", "São Paulo"],
                "symbols_&_quotes": ['"Special"', "A & B", "100% €"],
            }
        )
        csv_bytes = generate_csv_bytes(special_df)
        decoded = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(decoded.iloc[0]["col_ümlaut"], "München")
        self.assertEqual(decoded.iloc[2]["col_ümlaut"], "São Paulo")

    def test_excel_workbook_creation_and_sheet_names(self) -> None:
        """Verify Openpyxl creates multi-sheet workbook with expected analytical sheets."""
        excel_bytes = generate_excel_workbook_bytes(self.sample_df)
        self.assertIsInstance(excel_bytes, bytes)
        self.assertTrue(len(excel_bytes) > 0)

        # Inspect workbook with Openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), read_only=True)
        sheet_names = wb.sheetnames

        expected_sheets = [
            "Dataset",
            "Statistical Summary",
            "Data Quality",
            "Business KPIs",
            "Smart Insights",
        ]
        for sheet in expected_sheets:
            self.assertIn(sheet, sheet_names, f"Expected sheet '{sheet}' in Excel workbook")

        wb.close()

    def test_excel_long_text_and_missing_sections(self) -> None:
        """Verify workbook handles long text in insights and datasets without numeric columns."""
        text_df = pd.DataFrame(
            {
                "long_description": [
                    "This is an exceptionally long narrative description intended to test text wrapping and cell auto-fitting inside Openpyxl worksheets without causing formatting exceptions." * 3,
                    "Another test row with detailed context.",
                ],
                "category": ["Alpha", "Beta"],
            }
        )

        excel_bytes = generate_excel_workbook_bytes(text_df)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        self.assertIn("Statistical Summary", wb.sheetnames)
        self.assertIn("Dataset", wb.sheetnames)
        wb.close()

    def test_empty_dataset_handling(self) -> None:
        """Verify empty DataFrames do not cause unhandled exceptions in exporters."""
        empty_df = pd.DataFrame()

        # CSV export on empty
        csv_bytes = generate_csv_bytes(empty_df)
        self.assertEqual(csv_bytes, b"")

        # Excel export on empty
        excel_bytes = generate_excel_workbook_bytes(empty_df)
        self.assertIsInstance(excel_bytes, bytes)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        self.assertTrue(len(wb.sheetnames) > 0)
        wb.close()

    def test_pdf_availability_and_limitation(self) -> None:
        """Verify PDF availability check and limitation notice."""
        available = is_pdf_export_available()
        # In current environment, no PDF engine is installed
        self.assertFalse(available)

        msg = get_pdf_limitation_message()
        self.assertIn("reportlab", msg)
        self.assertIn("approve", msg.lower())

    def test_ui_empty_state_rendering(self) -> None:
        """Verify render_reports_view displays empty state when no dataset is loaded."""
        with patch("streamlit.session_state", {"dataset": None}), \
             patch("streamlit.markdown") as mock_markdown:

            render_reports_view()
            calls = [str(c) for c in mock_markdown.call_args_list]
            self.assertTrue(any("No Active Dataset to Export" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
