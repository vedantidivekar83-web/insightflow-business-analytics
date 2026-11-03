"""Unit test to verify sidebar status synchronization and filename formatting."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd


class TestSidebarDatasetSync(unittest.TestCase):
    """Test sidebar dataset status display and state transitions."""

    def test_sidebar_empty_state(self) -> None:
        """Verify sidebar status displays 'None loaded' when no dataset is in session state."""
        from src.ui.sidebar import render_sidebar

        mock_session_state = {"dataset_metadata": None}

        with patch("streamlit.session_state", mock_session_state), \
             patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.radio", return_value="🏠  Overview"):

            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)

            render_sidebar()

            markdown_calls = [str(call) for call in mock_markdown.call_args_list]
            found_none_loaded = any("None loaded" in call for call in markdown_calls)
            self.assertTrue(found_none_loaded, "Sidebar should show 'None loaded' when metadata is None")

    def test_sidebar_with_dataset_and_truncation(self) -> None:
        """Verify sidebar displays active dataset name and truncates long filenames."""
        from src.ui.sidebar import render_sidebar

        # 1. Short filename
        short_metadata = {
            "filename": "revenue.csv",
            "total_rows": 1250,
        }
        with patch("streamlit.session_state", {"dataset_metadata": short_metadata}), \
             patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.radio", return_value="🏠  Overview"):

            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)

            render_sidebar()
            calls = [str(c) for c in mock_markdown.call_args_list]
            self.assertTrue(any("revenue.csv" in c and "1,250 rows" in c for c in calls))

        # 2. Long filename truncation (> 20 characters)
        long_metadata = {
            "filename": "quarterly_enterprise_sales_report_2026.csv",
            "total_rows": 50000,
        }
        with patch("streamlit.session_state", {"dataset_metadata": long_metadata}), \
             patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.radio", return_value="🏠  Overview"):

            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)

            render_sidebar()
            calls = [str(c) for c in mock_markdown.call_args_list]
            # Should truncate to 17 chars + "..."
            self.assertTrue(any("quarterly_enterpr..." in c for c in calls))
            self.assertTrue(any("50,000 rows" in c for c in calls))

    def test_clear_dataset_restores_none_loaded(self) -> None:
        """Verify that clearing session state immediately restores 'None loaded'."""
        from src.ui.sidebar import render_sidebar

        state = {
            "dataset": pd.DataFrame({"a": [1, 2]}),
            "dataset_metadata": {"filename": "data.csv", "total_rows": 2},
        }

        # Clear state
        state["dataset"] = None
        state["dataset_metadata"] = None

        with patch("streamlit.session_state", state), \
             patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.radio", return_value="🏠  Overview"):

            mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
            mock_sidebar.__exit__ = MagicMock(return_value=None)

            render_sidebar()
            calls = [str(c) for c in mock_markdown.call_args_list]
            self.assertTrue(any("None loaded" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
