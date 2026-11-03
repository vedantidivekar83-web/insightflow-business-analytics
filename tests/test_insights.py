"""Unit test suite for Stage 7: Smart Insights Engine.

Tests data quality insights (missing values, duplicates), numerical outlier
and variance patterns, categorical concentration, correlation observations,
time-series trajectories, small datasets, and empty states.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.analytics.insights import (
    generate_smart_insights,
    generate_quality_insights,
    generate_numerical_insights,
    generate_categorical_insights,
    generate_correlation_insights,
    generate_time_insights,
    generate_business_insights,
)
from src.ui.views.insights import render_insights_view


class TestSmartInsightsEngine(unittest.TestCase):
    """Test suite covering the Smart Insights generator functions."""

    def test_quality_insights_duplicates_and_missing(self) -> None:
        """Verify generation of duplicate and high-missingness quality alerts."""
        df_dirty = pd.DataFrame(
            {
                "id": [1, 2, 1, 4, 5, 6, 7, 8, 9, 10],  # row 0 and 2 are dups
                "feature_missing": [np.nan, np.nan, np.nan, np.nan, 10, 20, 30, 40, 50, 60],  # 40% missing
                "clean_col": [1, 2, 1, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        insights = generate_quality_insights(df_dirty)
        cat_ids = [i["id"] for i in insights]

        self.assertIn("qual_duplicate_rows", cat_ids)
        self.assertIn("qual_missing_feature_missing", cat_ids)

        # Check evidence content
        dup_insight = next(i for i in insights if i["id"] == "qual_duplicate_rows")
        self.assertEqual(dup_insight["evidence"]["Duplicate Rows"], "1")
        self.assertTrue(len(dup_insight["why_it_matters"]) > 10)

    def test_quality_insights_clean_dataset(self) -> None:
        """Verify generation of positive integrity insight when dataset is clean."""
        df_clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        insights = generate_quality_insights(df_clean)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["severity"], "Positive")
        self.assertEqual(insights[0]["id"], "qual_clean_dataset")

    def test_numerical_outliers_and_dispersion(self) -> None:
        """Verify statistical outlier detection and high variance insights."""
        # Dataset with a clear outlier (999.0)
        df_nums = pd.DataFrame(
            {
                "regular": [10.0, 11.0, 10.5, 12.0, 11.5, 10.0, 11.0, 10.5, 999.0],
            }
        )

        insights = generate_numerical_insights(df_nums)
        ids = [i["id"] for i in insights]
        self.assertIn("num_outliers_regular", ids)

        outlier_insight = next(i for i in insights if i["id"] == "num_outliers_regular")
        self.assertIn("Outlier Count", outlier_insight["evidence"])

    def test_categorical_imbalance(self) -> None:
        """Verify detection of dominant categories (> 60% share)."""
        df_cats = pd.DataFrame(
            {
                "country": ["USA"] * 8 + ["Canada", "Mexico"],  # 80% USA
            }
        )

        insights = generate_categorical_insights(df_cats)
        self.assertTrue(len(insights) > 0)
        dom_insight = insights[0]
        self.assertIn("Dominant Segment", dom_insight["evidence"])
        self.assertEqual(dom_insight["evidence"]["Dominant Segment"], "USA")

    def test_correlation_insights(self) -> None:
        """Verify strong correlation detection and non-causality phrasing."""
        # Perfectly correlated columns
        x = [10.0, 20.0, 30.0, 40.0, 50.0]
        y = [20.0, 40.0, 60.0, 80.0, 100.0]  # r = 1.0
        df_corr = pd.DataFrame({"x": x, "y": y})

        insights = generate_correlation_insights(df_corr)
        self.assertEqual(len(insights), 1)
        corr_item = insights[0]
        self.assertIn("Strong Positive Correlation", corr_item["title"])
        self.assertIn("cause-and-effect", corr_item["why_it_matters"].lower())

    def test_time_series_trend(self) -> None:
        """Verify time-series trajectory direction and percentage shift."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        metrics = [100.0, 105.0, 110.0, 115.0, 120.0, 200.0, 210.0, 220.0, 230.0, 240.0]
        df_time = pd.DataFrame({"timestamp": dates, "sales": metrics})

        insights = generate_time_insights(df_time)
        self.assertEqual(len(insights), 1)
        trend_item = insights[0]
        self.assertIn("Upward Trend", trend_item["title"])
        self.assertIn("Shift Rate", trend_item["evidence"])

    def test_business_insights_sales_and_status(self) -> None:
        """Verify revenue throughput and negative status alerts."""
        df_biz = pd.DataFrame(
            {
                "quantity": [2, 3, 1, 4],
                "unit_price": [50.0, 100.0, 200.0, 25.0],
                "status": ["Cancelled", "Delivered", "Delivered", "Delivered"],  # 25% cancelled
            }
        )

        insights = generate_business_insights(df_biz)
        ids = [i["id"] for i in insights]

        self.assertIn("biz_inferred_revenue", ids)
        self.assertTrue(any("biz_status_negative" in i_id for i_id in ids))

    def test_small_and_empty_datasets(self) -> None:
        """Verify that small (n < 3) and completely empty datasets do not cause crashes."""
        empty_df = pd.DataFrame()
        small_df = pd.DataFrame({"a": [1, 2]})

        self.assertEqual(generate_smart_insights(empty_df), [])
        small_insights = generate_smart_insights(small_df)
        self.assertIsInstance(small_insights, list)

    def test_ui_empty_state_rendering(self) -> None:
        """Verify render_insights_view displays empty state banner when no dataset is loaded."""
        with patch("streamlit.session_state", {"dataset": None}), \
             patch("streamlit.markdown") as mock_markdown:

            render_insights_view()
            calls = [str(c) for c in mock_markdown.call_args_list]
            self.assertTrue(any("No Active Dataset for Smart Insights" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
