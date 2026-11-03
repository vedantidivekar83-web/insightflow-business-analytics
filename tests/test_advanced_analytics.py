"""Unit tests for the Advanced Analytics Engine (Stage 10).

Covers:
- Date column detection
- Trend analysis (daily, weekly, monthly, yearly, trajectory classification, missing dates)
- Comparative analysis (group summaries, secondary pivot cross-tabulations)
- Segmentation analysis (categorical and numerical quantile/range tiers)
- Performance rankings (scoring, tie handling, top/bottom performers, narrative explanation)
- Bivariate relationships (Pearson r, R-squared, regression slope & intercept)
- Edge cases: empty DataFrames, datasets without dates, datasets without numerical features, small datasets.
"""

import unittest
import pandas as pd
import numpy as np

from src.analytics.advanced import (
    detect_date_columns,
    compute_trend_analysis,
    compute_comparative_analysis,
    compute_categorical_segmentation,
    compute_numerical_segmentation,
    compute_performance_rankings,
    compute_bivariate_relationship,
)


class TestAdvancedAnalytics(unittest.TestCase):
    """Test suite for advanced analytics routines."""

    def setUp(self) -> None:
        """Create sample test datasets."""
        # 1. Standard business dataset with temporal, categorical, and numerical features
        self.df_standard = pd.DataFrame(
            {
                "OrderDate": [
                    "2023-01-05", "2023-01-20", "2023-02-14", "2023-02-28",
                    "2023-03-10", "2023-03-25", "2023-04-12", "2023-04-30",
                    "2023-05-15", "2023-06-01", "2023-06-20", "2023-07-04",
                ],
                "Region": [
                    "North", "South", "East", "West",
                    "North", "South", "East", "West",
                    "North", "South", "East", "West",
                ],
                "Category": [
                    "Hardware", "Software", "Hardware", "Services",
                    "Software", "Hardware", "Services", "Software",
                    "Hardware", "Software", "Services", "Hardware",
                ],
                "Revenue": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0],
                "Profit": [10.0, 15.0, 25.0, 30.0, 40.0, 45.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            }
        )

        # 2. Dataset with decreasing trend
        self.df_decreasing = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
                "Metric": [1000.0, 750.0, 500.0, 200.0],
            }
        )

        # 3. Dataset with stable trend
        self.df_stable = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
                "Metric": [100.0, 101.0, 99.5, 100.5],
            }
        )

        # 4. Dataset with missing and invalid dates
        self.df_missing_dates = pd.DataFrame(
            {
                "Date": ["2023-01-01", "invalid_date", None, "2023-02-01", "2023-03-01", "not_a_date"],
                "Value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        # 5. Dataset with ties in performance
        self.df_ties = pd.DataFrame(
            {
                "Department": ["Sales", "Engineering", "Marketing", "HR", "Support"],
                "Score": [100.0, 200.0, 200.0, 50.0, 100.0],
            }
        )

        # 6. Edge case datasets
        self.df_empty = pd.DataFrame()
        self.df_no_dates = pd.DataFrame({"Category": ["A", "B", "C"], "Val": [1, 2, 3]})
        self.df_no_nums = pd.DataFrame({"Category": ["A", "B"], "Status": ["Active", "Closed"]})
        self.df_single_row = pd.DataFrame({"Date": ["2023-01-01"], "Category": ["A"], "Val": [100]})

    # -------------------------------------------------------------------------
    # Date Column Detection
    # -------------------------------------------------------------------------
    def test_detect_date_columns_standard(self) -> None:
        """Verify date column detection identifies ISO date strings."""
        detected = detect_date_columns(self.df_standard)
        self.assertIn("OrderDate", detected)

    def test_detect_date_columns_native_datetime(self) -> None:
        """Verify native pandas datetime dtype is detected."""
        df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=5)})
        detected = detect_date_columns(df)
        self.assertEqual(detected, ["ts"])

    def test_detect_date_columns_empty(self) -> None:
        """Verify empty DataFrame returns empty date list."""
        self.assertEqual(detect_date_columns(self.df_empty), [])

    def test_detect_date_columns_no_dates(self) -> None:
        """Verify non-date columns are not flagged as dates."""
        detected = detect_date_columns(self.df_no_dates)
        self.assertEqual(detected, [])

    # -------------------------------------------------------------------------
    # Trend Analysis Tests
    # -------------------------------------------------------------------------
    def test_trend_analysis_increasing(self) -> None:
        """Verify that monotonic growth produces an 'Increasing' trend."""
        res = compute_trend_analysis(
            df=self.df_standard,
            date_col="OrderDate",
            metric_col="Revenue",
            frequency="M",
            agg_func="sum",
        )
        self.assertEqual(res["status"], "ok")
        summary = res["summary"]
        self.assertEqual(summary["trend_direction"], "Increasing")
        self.assertGreater(summary["overall_growth_pct"], 0.0)
        self.assertEqual(summary["period_count"], 7)
        self.assertEqual(len(res["timeline_df"]), 7)

    def test_trend_analysis_decreasing(self) -> None:
        """Verify that falling metric produces a 'Decreasing' trend."""
        res = compute_trend_analysis(
            df=self.df_decreasing,
            date_col="Date",
            metric_col="Metric",
            frequency="M",
            agg_func="sum",
        )
        self.assertEqual(res["status"], "ok")
        summary = res["summary"]
        self.assertEqual(summary["trend_direction"], "Decreasing")
        self.assertLess(summary["overall_growth_pct"], 0.0)

    def test_trend_analysis_stable(self) -> None:
        """Verify that negligible change produces a 'Stable' trend."""
        res = compute_trend_analysis(
            df=self.df_stable,
            date_col="Date",
            metric_col="Metric",
            frequency="M",
            agg_func="sum",
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["summary"]["trend_direction"], "Stable")

    def test_trend_analysis_frequencies(self) -> None:
        """Verify Daily, Weekly, Monthly, and Yearly groupings."""
        for freq in ["D", "W", "M"]:
            res = compute_trend_analysis(
                df=self.df_standard,
                date_col="OrderDate",
                metric_col="Revenue",
                frequency=freq,
            )
            self.assertEqual(res["status"], "ok")
            self.assertGreater(len(res["timeline_df"]), 1)

        # Yearly with multi-year data
        df_multi_year = pd.DataFrame(
            {"Date": ["2021-01-01", "2022-01-01", "2023-01-01"], "Val": [100.0, 200.0, 300.0]}
        )
        res_y = compute_trend_analysis(df_multi_year, "Date", "Val", frequency="Y")
        self.assertEqual(res_y["status"], "ok")
        self.assertEqual(res_y["summary"]["period_count"], 3)

    def test_trend_analysis_missing_and_invalid_dates(self) -> None:
        """Verify robust handling of invalid and null dates."""
        res = compute_trend_analysis(
            df=self.df_missing_dates,
            date_col="Date",
            metric_col="Value",
            frequency="M",
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["summary"]["period_count"], 3)  # Jan, Feb, Mar

    def test_trend_analysis_empty_and_insufficient(self) -> None:
        """Verify graceful error reporting when observations are insufficient."""
        res_empty = compute_trend_analysis(self.df_empty, "Date", "Val")
        self.assertEqual(res_empty["status"], "error")

        res_single = compute_trend_analysis(self.df_single_row, "Date", "Val")
        self.assertEqual(res_single["status"], "error")

    # -------------------------------------------------------------------------
    # Comparative Analysis Tests
    # -------------------------------------------------------------------------
    def test_comparative_analysis_single_group(self) -> None:
        """Verify group-wise aggregations and share calculations."""
        res = compute_comparative_analysis(
            df=self.df_standard,
            group_col="Region",
            metric_col="Revenue",
        )
        self.assertEqual(res["status"], "ok")
        tbl = res["summary_table"]
        self.assertEqual(len(tbl), 4)  # 4 regions
        self.assertIn("Share (%)", tbl.columns)
        self.assertAlmostEqual(tbl["Share (%)"].sum(), 100.0, places=1)
        self.assertEqual(res["metadata"]["group_count"], 4)
        self.assertIsNotNone(res["metadata"]["top_group"]["name"])

    def test_comparative_analysis_secondary_pivot(self) -> None:
        """Verify cross-tabulation when secondary dimension is provided."""
        res = compute_comparative_analysis(
            df=self.df_standard,
            group_col="Region",
            metric_col="Revenue",
            secondary_col="Category",
        )
        self.assertEqual(res["status"], "ok")
        self.assertFalse(res["pivot_table"].empty)
        self.assertEqual(len(res["pivot_table"]), 4)

    def test_comparative_analysis_empty(self) -> None:
        """Verify empty dataset comparative handling."""
        res = compute_comparative_analysis(self.df_empty, "Region", "Revenue")
        self.assertEqual(res["status"], "error")

    # -------------------------------------------------------------------------
    # Segmentation Analysis Tests
    # -------------------------------------------------------------------------
    def test_categorical_segmentation(self) -> None:
        """Verify categorical segmentation sizes and dominant segment identification."""
        res = compute_categorical_segmentation(
            df=self.df_standard,
            segment_col="Category",
            metric_col="Revenue",
        )
        self.assertEqual(res["status"], "ok")
        table = res["segment_table"]
        self.assertEqual(len(table), 3)  # Hardware, Software, Services
        self.assertEqual(res["metadata"]["dominant_segment"]["name"], "Hardware")
        self.assertEqual(res["metadata"]["total_records"], 12)

    def test_numerical_segmentation_quantiles(self) -> None:
        """Verify quantile-based tier creation."""
        res = compute_numerical_segmentation(
            df=self.df_standard,
            num_col="Revenue",
            method="quantiles",
            num_bins=4,
        )
        self.assertEqual(res["status"], "ok")
        table = res["segment_table"]
        self.assertEqual(len(table), 4)
        self.assertIn("Range", table.columns)
        self.assertIn("Share (%)", table.columns)

    def test_numerical_segmentation_ranges(self) -> None:
        """Verify equal-width range tier creation."""
        res = compute_numerical_segmentation(
            df=self.df_standard,
            num_col="Revenue",
            method="ranges",
            num_bins=3,
        )
        self.assertEqual(res["status"], "ok")
        table = res["segment_table"]
        self.assertEqual(len(table), 3)

    def test_numerical_segmentation_single_value(self) -> None:
        """Verify error when column lacks variation for binning."""
        df_const = pd.DataFrame({"Val": [10.0, 10.0, 10.0]})
        res = compute_numerical_segmentation(df_const, "Val", method="quantiles")
        self.assertEqual(res["status"], "error")

    # -------------------------------------------------------------------------
    # Performance Rankings Tests
    # -------------------------------------------------------------------------
    def test_performance_rankings_standard(self) -> None:
        """Verify ranking calculation, top/bottom isolation, and narrative explanation."""
        res = compute_performance_rankings(
            df=self.df_standard,
            group_col="Region",
            metric_col="Revenue",
            agg_func="sum",
            top_n=2,
        )
        self.assertEqual(res["status"], "ok")
        ranked = res["ranked_table"]
        self.assertEqual(ranked.iloc[0]["Rank"], 1)
        self.assertEqual(len(res["top_performers"]), 2)
        self.assertIn("ranked by SUM", res["metadata"]["explanation"])

    def test_performance_rankings_with_ties(self) -> None:
        """Verify that tied entities receive identical minimum ranks."""
        res = compute_performance_rankings(
            df=self.df_ties,
            group_col="Department",
            metric_col="Score",
            agg_func="sum",
            top_n=5,
        )
        self.assertEqual(res["status"], "ok")
        ranked = res["ranked_table"]

        # Engineering and Marketing are tied at 200.0, so both should have Rank 1
        rank_1_items = ranked[ranked["Rank"] == 1]
        self.assertEqual(len(rank_1_items), 2)

        # Sales and Support are tied at 100.0, so both should have Rank 3 (since two tied at 1)
        rank_3_items = ranked[ranked["Rank"] == 3]
        self.assertEqual(len(rank_3_items), 2)

    def test_performance_rankings_empty(self) -> None:
        """Verify empty dataset ranking error handling."""
        res = compute_performance_rankings(self.df_empty, "Dept", "Score")
        self.assertEqual(res["status"], "error")

    # -------------------------------------------------------------------------
    # Bivariate Relationship Tests
    # -------------------------------------------------------------------------
    def test_bivariate_relationship_positive(self) -> None:
        """Verify positive correlation and linear regression parameters."""
        res = compute_bivariate_relationship(
            df=self.df_standard,
            x_col="Revenue",
            y_col="Profit",
        )
        self.assertEqual(res["status"], "ok")
        metrics = res["metrics"]
        self.assertGreater(metrics["pearson_r"], 0.95)
        self.assertGreater(metrics["r_squared"], 0.90)
        self.assertGreater(metrics["slope"], 0.0)
        self.assertIn("Profit =", metrics["equation"])

    def test_bivariate_relationship_zero_variance(self) -> None:
        """Verify error when one variable is constant."""
        df_flat = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [5, 5, 5, 5]})
        res = compute_bivariate_relationship(df_flat, "X", "Y")
        self.assertEqual(res["status"], "error")

    def test_bivariate_relationship_insufficient_data(self) -> None:
        """Verify error when fewer than 3 pairs exist."""
        df_small = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        res = compute_bivariate_relationship(df_small, "X", "Y")
        self.assertEqual(res["status"], "error")


if __name__ == "__main__":
    unittest.main()
