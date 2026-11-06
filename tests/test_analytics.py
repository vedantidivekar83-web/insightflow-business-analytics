"""Unit test suite for Stage 6: Business Analytics Engine.

Tests descriptive statistics, data quality audits, business KPI inference,
correlation analysis (including constant columns), visualization helpers,
non-numeric datasets, and empty states.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import (
    compute_data_quality_summary,
    compute_column_quality_table,
)
from src.analytics.kpi import (
    detect_business_columns,
    compute_business_kpis,
    _clean_header,
)
from src.analytics.correlation import (
    compute_correlation_matrix,
    get_top_correlations,
)
from src.visualization.charts import (
    plot_numerical_distribution,
    plot_categorical_frequency,
    plot_correlation_heatmap,
    plot_time_series_trend,
)
from src.ui.views.analytics import render_analytics_view


class TestAnalyticsEngine(unittest.TestCase):
    """Test suite covering the Business Analytics Engine."""

    def setUp(self) -> None:
        """Create sample sales and general datasets for testing."""
        # 1. Sales-oriented dataset
        self.sales_df = pd.DataFrame(
            {
                "order_id": [1001, 1002, 1003, 1004, 1005],
                "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
                "quantity": [2, 5, 1, 4, 2],
                "unit_price": [100.0, 50.0, 200.0, 75.0, 150.0],
                "status": ["Delivered", "Delivered", "Cancelled", "Delivered", "Pending"],
                "constant_feature": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        # 2. General non-sales dataset (HR / Operations)
        self.hr_df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "department": ["IT", "HR", "IT", "Finance"],
                "experience_years": [5.0, 3.0, 8.0, 2.0],
                "rating": [4.5, 3.8, 4.9, 4.0],
            }
        )

        # 3. Categorical-only dataset
        self.cat_df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "C"],
                "label": ["High", "Low", "Medium", "High"],
            }
        )

    def test_statistical_calculations(self) -> None:
        """Verify calculations of mean, median, mode, min, max, std, variance, and quartiles."""
        data = pd.DataFrame({"metric": [10.0, 20.0, 20.0, 30.0, 40.0, 50.0, np.nan]})
        stats = compute_numerical_statistics(data)

        self.assertIn("metric", stats.index)
        row = stats.loc["metric"]

        # Expected values on [10, 20, 20, 30, 40, 50]
        self.assertAlmostEqual(row["Mean"], 28.3333, places=3)
        self.assertEqual(row["Median"], 25.0)
        self.assertEqual(row["Mode"], 20.0)
        self.assertEqual(row["Min"], 10.0)
        self.assertEqual(row["Max"], 50.0)
        self.assertEqual(row["Non-Null Count"], 6)
        self.assertEqual(row["Missing Count"], 1)

        # Quartiles
        self.assertAlmostEqual(row["Q1 (25%)"], 20.0, places=1)
        self.assertAlmostEqual(row["Q3 (75%)"], 37.5, places=1)
        self.assertAlmostEqual(row["IQR"], 17.5, places=1)

    def test_missing_values_and_duplicates(self) -> None:
        """Verify data quality summary and column quality table metrics."""
        df_with_dups = pd.DataFrame(
            {
                "colA": ["x", "y", "x", np.nan],
                "colB": [1, 2, 1, 4],
            }
        )
        # Note: row 0 and row 2 are duplicates ("x", 1)

        summary = compute_data_quality_summary(df_with_dups)
        self.assertEqual(summary["total_rows"], 4)
        self.assertEqual(summary["total_columns"], 2)
        self.assertEqual(summary["total_cells"], 8)
        self.assertEqual(summary["missing_cells"], 1)
        self.assertEqual(summary["missing_pct"], 12.5)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["duplicate_pct"], 25.0)

        table = compute_column_quality_table(df_with_dups)
        self.assertEqual(len(table), 2)
        col_a_row = table[table["Column"] == "colA"].iloc[0]
        self.assertEqual(col_a_row["Missing Count"], 1)
        self.assertEqual(col_a_row["Completeness (%)"], 75.0)
        self.assertEqual(col_a_row["Unique Values"], 2)

    def test_correlation_analysis_with_constant_columns(self) -> None:
        """Verify Pearson correlation calculation and graceful dropping of constant columns."""
        corr_matrix, dropped_cols = compute_correlation_matrix(self.sales_df)

        # constant_feature has std == 0 and must be dropped
        self.assertIn("constant_feature", dropped_cols)
        self.assertNotIn("constant_feature", corr_matrix.columns)

        # Between quantity and unit_price
        self.assertIn("quantity", corr_matrix.columns)
        self.assertIn("unit_price", corr_matrix.columns)
        self.assertEqual(corr_matrix.loc["quantity", "quantity"], 1.0)

        # Pairwise correlation extraction
        pairs = get_top_correlations(corr_matrix, top_n=3)
        self.assertTrue(len(pairs) > 0)
        self.assertIn("correlation", pairs[0])
        self.assertIn("relationship", pairs[0])

    def test_business_kpis_sales_dataset(self) -> None:
        """Verify inference of revenue, AOV, and status breakdown in sales datasets."""
        kpi_res = compute_business_kpis(self.sales_df)
        inferred = kpi_res["inferred_kpis"]

        # quantity: [2, 5, 1, 4, 2], unit_price: [100, 50, 200, 75, 150]
        # revenue = 200 + 250 + 200 + 300 + 300 = 1250
        self.assertTrue(any("Revenue" in k["name"] for k in inferred))
        rev_kpi = next(k for k in inferred if "Revenue" in k["name"])
        self.assertEqual(rev_kpi["value"], 1250.0)
        self.assertIn("Inferred", rev_kpi["assumption"])

        # Status distribution
        status_df = kpi_res["status_distribution"]
        self.assertIsNotNone(status_df)
        delivered_count = status_df[status_df["Status"] == "Delivered"].iloc[0]["Count"]
        self.assertEqual(delivered_count, 3)

    def test_business_kpis_currency_formatted_strings(self) -> None:
        """Verify inference when quantity and unit price contain currency symbols and commas."""
        currency_df = pd.DataFrame(
            {
                "Quantity (Units)": [5, 10, 2],
                "Unit Price ($)": ["$50.00", "$100.00", "$25.50"],
            }
        )
        res = compute_business_kpis(currency_df)
        inferred = res["inferred_kpis"]
        self.assertEqual(len(inferred), 2)
        rev_kpi = next(k for k in inferred if "Revenue" in k["name"])
        # (5*50) + (10*100) + (2*25.50) = 250 + 1000 + 51 = 1301.0
        self.assertEqual(rev_kpi["value"], 1301.0)
        self.assertEqual(rev_kpi["formatted"], "$1,301.00")

    def test_business_kpis_synonym_headers(self) -> None:
        """Verify matching of business column synonyms (e.g., Order Quantity and Price Each)."""
        synonym_df = pd.DataFrame(
            {
                "Order Quantity": [10, 20],
                "Price Each": [15.0, 30.0],
            }
        )
        res = compute_business_kpis(synonym_df)
        inferred = res["inferred_kpis"]
        self.assertEqual(len(inferred), 2)
        rev_kpi = next(k for k in inferred if "Revenue" in k["name"])
        # (10*15) + (20*30) = 150 + 600 = 750
        self.assertEqual(rev_kpi["value"], 750.0)

    def test_business_kpis_direct_revenue_priority(self) -> None:
        """Verify that direct revenue column is prioritized without duplicating quantity x price."""
        both_df = pd.DataFrame(
            {
                "Quantity": [10, 20],
                "Unit_Price": [15.0, 30.0],
                "Revenue": [140.0, 580.0],  # discounted actual revenue
            }
        )
        res = compute_business_kpis(both_df)
        inferred = res["inferred_kpis"]
        self.assertEqual(len(inferred), 2)
        rev_kpi = next(k for k in inferred if k["name"] == "Total Revenue")
        # Direct revenue sum: 140 + 580 = 720
        self.assertEqual(rev_kpi["value"], 720.0)
        self.assertIn("Directly aggregated", rev_kpi["assumption"])

    def test_business_kpis_non_sales_dataset(self) -> None:
        """Verify fallback to primary numerical metrics on non-sales datasets with explicit note."""
        kpi_res = compute_business_kpis(self.hr_df)
        # Should not infer sales/revenue
        self.assertEqual(len(kpi_res["inferred_kpis"]), 0)

        # Should detect primary numerical metric (experience_years or rating)
        primary = kpi_res["primary_metric"]
        self.assertIsNotNone(primary)
        self.assertIn(primary["column_name"], ["experience_years", "rating"])
        self.assertIn("Primary", primary["assumption"])

    def test_dataset_without_numerical_columns(self) -> None:
        """Verify that datasets lacking numerical features do not cause crashes."""
        stats = compute_numerical_statistics(self.cat_df)
        self.assertTrue(stats.empty)

        corr_matrix, dropped = compute_correlation_matrix(self.cat_df)
        self.assertTrue(corr_matrix.empty)
        self.assertEqual(dropped, [])

    def test_empty_dataset_state(self) -> None:
        """Verify handling of completely empty DataFrames."""
        empty_df = pd.DataFrame()

        self.assertTrue(compute_numerical_statistics(empty_df).empty)

        quality = compute_data_quality_summary(empty_df)
        self.assertEqual(quality["total_rows"], 0)

        kpis = compute_business_kpis(empty_df)
        self.assertEqual(kpis["total_records"], 0)

        corr_matrix, dropped = compute_correlation_matrix(empty_df)
        self.assertTrue(corr_matrix.empty)

    def test_visualization_generation(self) -> None:
        """Verify Plotly figure factories produce valid figures without errors."""
        fig_num = plot_numerical_distribution(self.sales_df, "quantity")
        self.assertIsNotNone(fig_num)

        fig_cat = plot_categorical_frequency(self.sales_df, "status")
        self.assertIsNotNone(fig_cat)

        corr_matrix, _ = compute_correlation_matrix(self.sales_df)
        fig_corr = plot_correlation_heatmap(corr_matrix)
        self.assertIsNotNone(fig_corr)

        fig_ts = plot_time_series_trend(self.sales_df, "order_date", "quantity")
        self.assertIsNotNone(fig_ts)

    def test_ui_empty_state_rendering(self) -> None:
        """Verify render_analytics_view displays empty state banner when no dataset is present."""
        with patch("streamlit.session_state", {"dataset": None}), \
             patch("streamlit.markdown") as mock_markdown:

            render_analytics_view()
            calls = [str(c) for c in mock_markdown.call_args_list]
            self.assertTrue(any("No Active Dataset for Analytics" in c for c in calls))

    def test_unicode_normalization_header_matching(self) -> None:
        """Verify Unicode normalization strips Latin accents and maps headers to base ASCII."""
        self.assertEqual(_clean_header("Quantité"), "quantite")
        self.assertEqual(_clean_header("Coût Unitaire ($)"), "coutunitaire")
        self.assertEqual(_clean_header("Ventes (€)"), "ventes")
        self.assertEqual(_clean_header("Bénéfice Net"), "beneficenet")
        self.assertEqual(_clean_header("Numéro de Commande"), "numerodecommande")
        self.assertEqual(_clean_header("Año"), "ano")
        self.assertEqual(_clean_header("État"), "etat")
        self.assertEqual(_clean_header("Chiffre d'Affaires"), "chiffredaffaires")

    def test_business_kpis_accented_international_headers(self) -> None:
        """Verify business KPI engine recognizes accented international headers and infers metrics."""
        intl_df = pd.DataFrame(
            {
                "Quantité": [10, 20],
                "Coût Unitaire ($)": [15.0, 30.0],
                "Statut": ["Livré", "En cours"],
            }
        )
        res = compute_business_kpis(intl_df)
        detected = res["detected_columns"]
        self.assertEqual(detected["quantity"], "Quantité")
        self.assertEqual(detected["unit_price"], "Coût Unitaire ($)")
        self.assertEqual(detected["status"], "Statut")

        inferred = res["inferred_kpis"]
        self.assertEqual(len(inferred), 2)
        rev_kpi = next(k for k in inferred if "Revenue" in k["name"])
        # (10*15) + (20*30) = 150 + 600 = 750
        self.assertEqual(rev_kpi["value"], 750.0)

        aov_kpi = next(k for k in inferred if "Order Value" in k["name"])
        self.assertEqual(aov_kpi["value"], 375.0)

        # Status distribution computed
        status_dist = res["status_distribution"]
        self.assertIsNotNone(status_dist)
        self.assertEqual(len(status_dist), 2)


if __name__ == "__main__":
    unittest.main()
