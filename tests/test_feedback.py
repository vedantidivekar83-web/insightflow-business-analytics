"""Unit tests for the Feedback & Suggestions component."""

import unittest
from unittest.mock import MagicMock, patch
import streamlit as st

from src.ui.feedback import save_feedback_entry, render_feedback_section
from src.ui.views.overview import render_overview_view


class TestFeedbackComponent(unittest.TestCase):
    """Test suite for user feedback and suggestions module."""

    def setUp(self):
        """Reset mock session state before each test."""
        self.mock_state = {
            "feedback_submitted": False,
            "feedback_history": [],
            "dataset": None,
            "dataset_metadata": None,
        }

    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    def test_save_feedback_entry_structure_and_no_pii(self, mock_dirs, mock_open):
        """Verify feedback entry structure only stores rating, suggestions, and timestamp."""
        with patch.dict(st.session_state, self.mock_state, clear=True):
            entry = {
                "timestamp": "2026-11-07T10:00:00Z",
                "rating": 5,
                "suggestions": "Great tool! Would love to see cohort analysis.",
            }
            save_feedback_entry(entry)

            self.assertEqual(len(st.session_state["feedback_history"]), 1)
            saved = st.session_state["feedback_history"][0]
            self.assertEqual(saved["rating"], 5)
            self.assertEqual(saved["suggestions"], "Great tool! Would love to see cohort analysis.")
            self.assertIn("timestamp", saved)

            # Ensure no PII fields exist
            for forbidden_key in ["name", "email", "username", "dataset", "data", "ip"]:
                self.assertNotIn(forbidden_key, saved)

    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    def test_save_feedback_entry_optional_suggestion_none(self, mock_dirs, mock_open):
        """Verify that suggestions can be None when user only submits a star rating."""
        with patch.dict(st.session_state, self.mock_state, clear=True):
            entry = {
                "timestamp": "2026-11-07T10:05:00Z",
                "rating": 4,
                "suggestions": None,
            }
            save_feedback_entry(entry)

            self.assertEqual(len(st.session_state["feedback_history"]), 1)
            saved = st.session_state["feedback_history"][0]
            self.assertEqual(saved["rating"], 4)
            self.assertIsNone(saved["suggestions"])

    def test_render_feedback_section_unsubmitted_clean(self):
        """Verify render_feedback_section executes cleanly in default unsubmitted state."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("streamlit.feedback", return_value=None), \
             patch("streamlit.text_area", return_value=""), \
             patch("streamlit.button", return_value=False):
            # Should run without raising any exceptions
            render_feedback_section()
            self.assertFalse(st.session_state["feedback_submitted"])
            self.assertEqual(st.session_state["feedback_history"], [])

    def test_render_feedback_section_submitted_success_state(self):
        """Verify render_feedback_section renders success state when feedback_submitted is True."""
        state_submitted = {
            "feedback_submitted": True,
            "feedback_history": [{"timestamp": "2026-11-07T10:00:00Z", "rating": 5, "suggestions": "Nice!"}],
        }
        with patch.dict(st.session_state, state_submitted, clear=True), \
             patch("streamlit.success") as mock_success, \
             patch("streamlit.button", return_value=False):
            render_feedback_section()
            mock_success.assert_called_once()
            self.assertTrue(st.session_state["feedback_submitted"])

    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    def test_render_feedback_submission_triggers_save(self, mock_dirs, mock_open):
        """Verify submitting with star rating triggers save_feedback_entry."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("streamlit.feedback", return_value=3), \
             patch("streamlit.text_area", return_value="Add dark mode toggle"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            self.assertTrue(st.session_state["feedback_submitted"])
            self.assertEqual(len(st.session_state["feedback_history"]), 1)
            saved = st.session_state["feedback_history"][0]
            self.assertEqual(saved["rating"], 4)  # star index 3 + 1 = 4 stars
            self.assertEqual(saved["suggestions"], "Add dark mode toggle")
            mock_rerun.assert_called_once()

    def test_render_feedback_warns_when_star_not_selected(self):
        """Verify warning is shown if Submit is clicked without selecting stars."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("streamlit.feedback", return_value=None), \
             patch("streamlit.text_area", return_value="Just a suggestion"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.warning") as mock_warning, \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            mock_warning.assert_called_once()
            self.assertFalse(st.session_state["feedback_submitted"])
            self.assertEqual(len(st.session_state["feedback_history"]), 0)
            mock_rerun.assert_not_called()

    def test_overview_view_includes_feedback_section(self):
        """Verify Overview page renders cleanly with feedback section integrated."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("src.ui.views.overview.render_feedback_section") as mock_feedback_render:
            render_overview_view()
            mock_feedback_render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
