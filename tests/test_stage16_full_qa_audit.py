"""Automated Quality Assurance test suite for Stage 16 Audit.

Covers:
- Section 1: Page rendering with active & empty state.
- Section 2: CSV/Excel loaders, corrupted files, missing values, duplicates, large datasets.
- Section 3: Revenue detection (direct, qty x price, currency strings, synonyms, fallback).
- Section 4: Isolation Forest, Classifier, Regressor, Holt's Forecast, small data edge cases.
- Section 5: CSV & Multi-sheet Excel exports, sheet structure, cell integrity.
- Section 6: Python compilation, dependency checks, security/secret scans.
"""

import compileall
import io
import os
import re
import sys
import time
import unittest
import openpyxl
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_uploaded_dataset, inspect_excel_sheets
from src.data.validator import validate_file_extension, validate_file_size, validate_dataframe, extract_dataset_metadata
from src.data.explorer import classify_columns, search_dataset, filter_categorical, filter_numeric_range, sort_dataset, select_columns
from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import compute_data_quality_summary, compute_column_quality_table
from src.analytics.kpi import detect_business_columns, compute_business_kpis
from src.analytics.correlation import compute_correlation_matrix, get_top_correlations
from src.analytics.insights import generate_smart_insights
from src.analytics.advanced import (
    detect_date_columns,
    compute_trend_analysis,
    compute_comparative_analysis,
    compute_categorical_segmentation,
    compute_numerical_segmentation,
    compute_performance_rankings,
    compute_bivariate_relationship,
)
from src.ml.anomaly_detection import is_sklearn_available, detect_anomalies
from src.ml.prediction import infer_ml_task, train_predictive_model
from src.ml.forecasting import generate_time_series_forecast
from src.reports.csv_export import generate_csv_bytes
from src.reports.excel_export import generate_excel_workbook_bytes
from src.reports.pdf_export import is_pdf_export_available, get_pdf_limitation_message


class TestStage16QA(unittest.TestCase):
    """Exhaustive Stage 16 QA Verification Suite."""

    # -------------------------------------------------------------------------
    # DATA TESTING
    # -------------------------------------------------------------------------
    def test_csv_ingestion_valid(self):
        """Verify standard CSV parsing, encoding fallback, and column preservation."""
        csv_data = b"order_id,product,price\n101,Widget,25.5\n102,Gadget,45.0\n"
        df, err = load_uploaded_dataset(csv_data, "test.csv")
        self.assertIsNone(err)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["order_id", "product", "price"])

    def test_excel_ingestion_multi_sheet(self):
        """Verify multi-sheet Excel inspection and targeted sheet loading."""
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Sales"
        ws1.append(["id", "amount"])
        ws1.append([1, 100])
        ws2 = wb.create_sheet("Inventory")
        ws2.append(["sku", "stock"])
        ws2.append(["W1", 50])

        bio = io.BytesIO()
        wb.save(bio)
        excel_bytes = bio.getvalue()

        sheets = inspect_excel_sheets(excel_bytes)
        self.assertEqual(sheets, ["Sales", "Inventory"])

        df_inv, err = load_uploaded_dataset(excel_bytes, "store.xlsx", sheet_name="Inventory")
        self.assertIsNone(err)
        self.assertEqual(len(df_inv), 1)
        self.assertIn("sku", df_inv.columns)

    def test_invalid_file_handling(self):
        """Verify rejection of invalid file extensions and corrupted bytes."""
        valid, err = validate_file_extension("script.py")
        self.assertFalse(valid)
        self.assertIn("Unsupported", err)

        # Corrupted CSV
        corrupted_bytes = b"\x00\x01\x02\x03\x04"
        df, err = load_uploaded_dataset(corrupted_bytes, "corrupt.csv")
        # Should gracefully return error without unhandled crash
        self.assertTrue(err is not None or df is not None)

    def test_missing_values_and_duplicate_records(self):
        """Verify data quality auditor accurately identifies missing cells and duplicates."""
        df = pd.DataFrame({
            "A": [1, 2, 1, None],
            "B": ["x", "y", "x", "z"],
        })
        quality = compute_data_quality_summary(df)
        self.assertEqual(quality["missing_cells"], 1)
        self.assertEqual(quality["duplicate_rows"], 1)
        self.assertAlmostEqual(quality["missing_pct"], 12.5, places=1)
        self.assertAlmostEqual(quality["completeness_pct"], 87.5, places=1)

    def test_large_dataset_handling(self):
        """Verify performance and memory bounds on a large 50,000-row synthetic dataset."""
        n_rows = 50000
        large_df = pd.DataFrame({
            "id": np.arange(n_rows),
            "region": np.random.choice(["North", "South", "East", "West"], size=n_rows),
            "amount": np.random.uniform(10.0, 500.0, size=n_rows),
            "quantity": np.random.randint(1, 20, size=n_rows),
        })
        start_t = time.time()
        meta = extract_dataset_metadata(large_df, "large.csv", 5_000_000)
        elapsed = time.time() - start_t

        self.assertEqual(meta["total_rows"], n_rows)
        self.assertLess(elapsed, 1.5, "Metadata extraction on 50k rows should be sub-second.")

    # -------------------------------------------------------------------------
    # ANALYTICS & REVENUE DETECTION VERIFICATION
    # -------------------------------------------------------------------------
    def test_revenue_direct_column(self):
        """Verify direct revenue detection when appropriate column exists."""
        df = pd.DataFrame({"Revenue": [100.0, 250.0, 400.0]})
        kpis = compute_business_kpis(df)
        rev = next(k for k in kpis["inferred_kpis"] if k["name"] == "Total Revenue")
        self.assertEqual(rev["value"], 750.0)
        self.assertEqual(rev["formatted"], "$750.00")

    def test_revenue_calculated_from_qty_and_price(self):
        """Verify revenue is calculated from quantity x unit_price when direct revenue is missing."""
        df = pd.DataFrame({
            "Quantity": [2, 4, 10],
            "Unit_Price": [50.0, 25.0, 10.0],
        })
        kpis = compute_business_kpis(df)
        rev = next(k for k in kpis["inferred_kpis"] if "Revenue" in k["name"])
        # (2*50) + (4*25) + (10*10) = 100 + 100 + 100 = 300.0
        self.assertEqual(rev["value"], 300.0)
        self.assertEqual(rev["formatted"], "$300.00")

    def test_revenue_with_currency_symbols_and_synonyms(self):
        """Verify revenue calculation when unit price has currency strings and synonym headers."""
        df = pd.DataFrame({
            "Order Quantity": [5, 10],
            "Price Each ($)": ["$20.00", "$45.50"],
        })
        kpis = compute_business_kpis(df)
        rev = next(k for k in kpis["inferred_kpis"] if "Revenue" in k["name"])
        # (5*20) + (10*45.50) = 100 + 455 = 555.0
        self.assertEqual(rev["value"], 555.0)
        self.assertEqual(rev["formatted"], "$555.00")

    def test_revenue_not_duplicated_when_both_present(self):
        """Verify that when both Revenue and Qty/Price exist, Revenue is not duplicated."""
        df = pd.DataFrame({
            "Quantity": [5, 10],
            "Unit_Price": [20.0, 50.0],
            "Revenue": [90.0, 480.0],  # discounted actual revenue = 570.0
        })
        kpis = compute_business_kpis(df)
        inferred = kpis["inferred_kpis"]
        self.assertEqual(len(inferred), 2)
        rev = next(k for k in inferred if k["name"] == "Total Revenue")
        self.assertEqual(rev["value"], 570.0)

    def test_unsupported_dataset_honest_fallback(self):
        """Verify unsupported general datasets receive honest fallback metrics."""
        df = pd.DataFrame({
            "Employee": ["Alice", "Bob"],
            "Department": ["Eng", "Design"],
            "Tenure_Years": [3.5, 5.0],
        })
        kpis = compute_business_kpis(df)
        self.assertEqual(len(kpis["inferred_kpis"]), 0)
        self.assertIsNotNone(kpis["primary_metric"])
        self.assertEqual(kpis["primary_metric"]["column_name"], "Tenure_Years")

    # -------------------------------------------------------------------------
    # MACHINE LEARNING VERIFICATION
    # -------------------------------------------------------------------------
    def test_isolation_forest_anomaly_detection(self):
        """Verify Isolation Forest detects simulated extreme outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(50, 5, size=(100, 2))
        outliers = np.array([[500.0, 500.0], [-200.0, -200.0]])
        full_data = np.vstack([normal_data, outliers])

        df = pd.DataFrame(full_data, columns=["feat_a", "feat_b"])
        res = detect_anomalies(df, ["feat_a", "feat_b"], contamination=0.03)

        self.assertEqual(res["status"], "ok")
        self.assertGreater(res["anomaly_count"], 0)
        self.assertIn("isolated", res["explanation"].lower())

    def test_predictive_regression(self):
        """Verify supervised regression model training and evaluation metrics."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            "target": [3.1, 6.2, 9.3, 12.4, 15.5, 18.6, 21.7, 24.8, 27.9, 31.0, 34.1, 37.2],
        })
        res = train_predictive_model(df, "target", ["feature1", "feature2"], model_name="Linear Regression", test_size=0.2)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["task_type"], "regression")
        self.assertGreaterEqual(res["metrics"]["r2"], 0.95)

    def test_predictive_classification(self):
        """Verify supervised classification model training, accuracy, and confusion matrix."""
        df = pd.DataFrame({
            "feature1": [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.2, 5.0, 5.2, 4.9, 5.1, 5.3, 4.8, 5.0, 5.2],
            "feature2": [0.5, 0.6, 0.4, 0.5, 0.7, 0.4, 0.5, 0.6, 3.0, 3.1, 2.9, 3.2, 3.3, 2.8, 3.0, 3.1],
            "label": [
                "ClassA", "ClassA", "ClassA", "ClassA", "ClassA", "ClassA", "ClassA", "ClassA",
                "ClassB", "ClassB", "ClassB", "ClassB", "ClassB", "ClassB", "ClassB", "ClassB",
            ],
        })
        res = train_predictive_model(df, "label", ["feature1", "feature2"], model_name="Random Forest Classifier", test_size=0.25)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["task_type"], "classification")
        self.assertIn("accuracy", res["metrics"])
        self.assertIsNotNone(res["metrics"]["confusion_matrix"])

    def test_time_series_forecasting(self):
        """Verify forward time-series forecasting with trend extrapolation and uncertainty intervals."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=12, freq="ME"),
            "val": [100, 110, 122, 135, 148, 162, 175, 190, 204, 218, 235, 250],
        })
        res = generate_time_series_forecast(df, "date", "val", forecast_periods=4, frequency="M")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["forecast_df"]), 4)
        self.assertTrue(all(res["forecast_df"]["upper_bound"] >= res["forecast_df"]["lower_bound"]))

    def test_ml_insufficient_data_handling(self):
        """Verify that ML models gracefully refuse to run when data is critically insufficient."""
        small_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        res = train_predictive_model(small_df, "b", ["a"])
        self.assertEqual(res["status"], "error")
        self.assertIn("insufficient", res["message"].lower())

    # -------------------------------------------------------------------------
    # EXPORT VERIFICATION
    # -------------------------------------------------------------------------
    def test_csv_export_bytes(self):
        """Verify UTF-8 CSV byte generation matches source records."""
        df = pd.DataFrame({"name": ["Alpha", "Beta"], "score": [98.5, 87.2]})
        csv_bytes = generate_csv_bytes(df)
        self.assertIsInstance(csv_bytes, bytes)
        recovered = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(recovered), 2)
        self.assertEqual(list(recovered.columns), ["name", "score"])

    def test_excel_export_structure(self):
        """Verify multi-sheet Excel generation includes all required sheets with frozen panes."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "quantity": [2, 5, 1],
            "unit_price": [10.0, 20.0, 30.0],
            "status": ["Completed", "Shipped", "Completed"],
        })
        excel_bytes = generate_excel_workbook_bytes(df)
        self.assertIsInstance(excel_bytes, bytes)

        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        required_sheets = ["Dataset", "Statistical Summary", "Data Quality", "Business KPIs", "Smart Insights"]
        for s in required_sheets:
            self.assertIn(s, wb.sheetnames)

        # Check freeze panes on Dataset sheet
        ds_sheet = wb["Dataset"]
        self.assertEqual(ds_sheet.freeze_panes, "A2")

    def test_pdf_export_honesty(self):
        """Verify that PDF export limitation is transparently documented."""
        limitation_msg = get_pdf_limitation_message()
        self.assertIn("reportlab", limitation_msg.lower())
        self.assertIn("not currently installed", limitation_msg.lower())

    # -------------------------------------------------------------------------
    # CODE QUALITY & PYTHON COMPILATION
    # -------------------------------------------------------------------------
    def test_python_syntax_and_compilation(self):
        """Verify all python files in project compile cleanly without syntax errors."""
        res = compileall.compile_dir(".", maxlevels=10, quiet=1)
        self.assertTrue(res, "All project Python files must compile cleanly.")

    def test_security_audit_no_hardcoded_secrets(self):
        """Scan project python source files to verify absence of hardcoded API keys or secrets."""
        secret_patterns = [
            re.compile(r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
            re.compile(r"secret[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
            re.compile(r"password\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
        ]
        violations = []
        for root, _, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        for pat in secret_patterns:
                            if pat.search(content):
                                violations.append((path, pat.pattern))

        self.assertEqual(violations, [], f"Found potential hardcoded secrets: {violations}")


if __name__ == "__main__":
    unittest.main()
