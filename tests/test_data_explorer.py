"""Unit test suite for Stage 5: Interactive Data Explorer.

Tests search, categorical filtering, numerical filtering, sorting,
column projections, combined pipelines, and empty dataset states.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

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
from src.ui.views.explorer import render_explorer_view


class TestDataExplorer(unittest.TestCase):
    """Test suite covering data explorer engine and state handling."""

    def setUp(self) -> None:
        """Create sample dataset for testing transformations."""
        self.sample_df = pd.DataFrame(
            {
                "employee_id": [101, 102, 103, 104, 105],
                "name": ["Alice Smith", "Bob Jones", "Charlie Brown", "David Lee", "Eve Miller"],
                "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
                "salary": [115000.0, 72000.0, 125000.0, 85000.0, 68000.0],
                "performance_score": [4.8, 3.9, 4.9, 4.1, np.nan],
                "is_remote": [True, False, True, False, True],
            }
        )

    def test_classify_columns(self) -> None:
        """Verify proper classification of numerical and categorical columns."""
        classified = classify_columns(self.sample_df)
        self.assertIn("employee_id", classified["numerical"])
        self.assertIn("salary", classified["numerical"])
        self.assertIn("performance_score", classified["numerical"])
        self.assertIn("name", classified["categorical"])
        self.assertIn("department", classified["categorical"])
        self.assertIn("is_remote", classified["categorical"])

    def test_search_dataset(self) -> None:
        """Verify case-insensitive global search across all columns."""
        # Search by name (case-insensitive)
        result1 = search_dataset(self.sample_df, "alice")
        self.assertEqual(len(result1), 1)
        self.assertEqual(result1.iloc[0]["name"], "Alice Smith")

        # Search by numeric value string
        result2 = search_dataset(self.sample_df, "125000")
        self.assertEqual(len(result2), 1)
        self.assertEqual(result2.iloc[0]["name"], "Charlie Brown")

        # Search by partial string across multiple rows
        result3 = search_dataset(self.sample_df, "marketing")
        self.assertEqual(len(result3), 2)

        # Empty search returns original copy
        result_empty = search_dataset(self.sample_df, "")
        self.assertEqual(len(result_empty), len(self.sample_df))

        # Special characters search (safe, regex=False)
        result_special = search_dataset(self.sample_df, "[test]")
        self.assertEqual(len(result_special), 0)

    def test_categorical_filtering(self) -> None:
        """Verify filtering by one or more categorical values."""
        # Single category
        res_eng = filter_categorical(self.sample_df, "department", ["Engineering"])
        self.assertEqual(len(res_eng), 2)
        self.assertTrue(all(d == "Engineering" for d in res_eng["department"]))

        # Multiple categories
        res_multi = filter_categorical(self.sample_df, "department", ["Engineering", "Sales"])
        self.assertEqual(len(res_multi), 3)

        # Empty selection returns all
        res_all = filter_categorical(self.sample_df, "department", [])
        self.assertEqual(len(res_all), 5)

        # Distinct unique categories helper
        unique_depts = get_unique_categories(self.sample_df, "department")
        self.assertEqual(unique_depts, ["Engineering", "Marketing", "Sales"])

    def test_numerical_filtering(self) -> None:
        """Verify numerical range filtering."""
        bounds = get_numeric_bounds(self.sample_df, "salary")
        self.assertEqual(bounds, (68000.0, 125000.0))

        # Filter salary between 70k and 100k
        res_salary = filter_numeric_range(self.sample_df, "salary", 70000.0, 100000.0)
        self.assertEqual(len(res_salary), 2)  # Bob (72k) and David (85k)
        self.assertListEqual(list(res_salary["name"]), ["Bob Jones", "David Lee"])

        # Range filtering on column containing NaN
        res_perf = filter_numeric_range(self.sample_df, "performance_score", 4.0, 5.0)
        self.assertEqual(len(res_perf), 3)  # Alice, Charlie, David
        self.assertFalse(res_perf["performance_score"].isna().any())

    def test_sorting(self) -> None:
        """Verify sorting ascending and descending with NaNs placed last."""
        # Sort ascending
        res_asc = sort_dataset(self.sample_df, "salary", ascending=True)
        self.assertEqual(res_asc.iloc[0]["salary"], 68000.0)
        self.assertEqual(res_asc.iloc[-1]["salary"], 125000.0)

        # Sort descending
        res_desc = sort_dataset(self.sample_df, "salary", ascending=False)
        self.assertEqual(res_desc.iloc[0]["salary"], 125000.0)
        self.assertEqual(res_desc.iloc[-1]["salary"], 68000.0)

        # Sort with NaNs
        res_nan = sort_dataset(self.sample_df, "performance_score", ascending=True)
        self.assertTrue(pd.isna(res_nan.iloc[-1]["performance_score"]))

    def test_column_selection(self) -> None:
        """Verify column projection does not mutate original DataFrame."""
        projected = select_columns(self.sample_df, ["name", "salary"])
        self.assertEqual(list(projected.columns), ["name", "salary"])
        self.assertEqual(len(projected), 5)

        # Original remains completely intact
        self.assertEqual(len(self.sample_df.columns), 6)

    def test_multiple_filters_combined(self) -> None:
        """Verify search, categorical, numerical and sorting combined in a pipeline."""
        df = self.sample_df.copy()

        # Step 1: Filter to Engineering
        df = filter_categorical(df, "department", ["Engineering"])
        self.assertEqual(len(df), 2)

        # Step 2: Numerical range salary >= 120,000
        df = filter_numeric_range(df, "salary", 120000.0, 150000.0)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["name"], "Charlie Brown")

        # Step 3: Column selection
        df = select_columns(df, ["name", "salary"])
        self.assertEqual(list(df.columns), ["name", "salary"])

    def test_empty_dataset_handling(self) -> None:
        """Verify engine safely handles empty DataFrame."""
        empty_df = pd.DataFrame()

        self.assertTrue(search_dataset(empty_df, "test").empty)
        self.assertTrue(filter_categorical(empty_df, "col", ["a"]).empty)
        self.assertTrue(filter_numeric_range(empty_df, "col", 0.0, 10.0).empty)
        self.assertTrue(sort_dataset(empty_df, "col").empty)
        self.assertTrue(select_columns(empty_df, ["col"]).empty)

    def test_empty_state_in_ui(self) -> None:
        """Verify render_explorer_view shows empty state when no dataset is in session state."""
        with patch("streamlit.session_state", {"dataset": None, "explorer_version": 0}), \
             patch("streamlit.markdown") as mock_markdown:

            render_explorer_view()

            calls = [str(call) for call in mock_markdown.call_args_list]
            self.assertTrue(any("No Active Dataset to Explore" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
