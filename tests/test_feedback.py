"""Unit tests for the Supabase Feedback & Suggestions integration."""

import unittest
from unittest.mock import MagicMock, patch
import requests
import streamlit as st

from src.ui.feedback import (
    get_supabase_credentials,
    submit_feedback_to_supabase,
    render_feedback_section,
)
from src.ui.views.overview import render_overview_view


class TestSupabaseFeedbackIntegration(unittest.TestCase):
    """Test suite for Supabase feedback database integration."""

    def setUp(self):
        """Reset mock session state before each test."""
        self.mock_state = {
            "feedback_submitted": False,
            "dataset": None,
            "dataset_metadata": None,
        }

    # -------------------------------------------------------------------------
    # 1. Credentials Extraction & Safety Tests
    # -------------------------------------------------------------------------
    def test_get_supabase_credentials_success(self):
        """Verify valid Supabase credentials are extracted from st.secrets."""
        fake_secrets = {
            "supabase": {
                "url": "https://testproject.supabase.co",
                "publishable_key": "sb_publishable_anon_key_12345",
            }
        }
        with patch("src.ui.feedback.st.secrets", fake_secrets):
            url, key = get_supabase_credentials()
            self.assertEqual(url, "https://testproject.supabase.co")
            self.assertEqual(key, "sb_publishable_anon_key_12345")

    def test_get_supabase_credentials_missing_section(self):
        """Verify None is returned when [supabase] section is absent."""
        fake_secrets = {"other_service": {"key": "value"}}
        with patch("src.ui.feedback.st.secrets", fake_secrets):
            url, key = get_supabase_credentials()
            self.assertIsNone(url)
            self.assertIsNone(key)

    def test_get_supabase_credentials_empty_or_missing_keys(self):
        """Verify None is returned when url or publishable_key is empty."""
        fake_secrets = {"supabase": {"url": "", "publishable_key": None}}
        with patch("src.ui.feedback.st.secrets", fake_secrets):
            url, key = get_supabase_credentials()
            self.assertIsNone(url)
            self.assertIsNone(key)

    def test_get_supabase_credentials_secrets_exception(self):
        """Verify local development safety when st.secrets raises an exception."""
        mock_sec = MagicMock()
        mock_sec.__contains__.side_effect = Exception("Secrets file missing")
        with patch("src.ui.feedback.st.secrets", mock_sec):
            url, key = get_supabase_credentials()
            self.assertIsNone(url)
            self.assertIsNone(key)

    # -------------------------------------------------------------------------
    # 2. Database Insertion (PostgREST HTTP API) Tests
    # -------------------------------------------------------------------------
    @patch("requests.post")
    def test_submit_feedback_to_supabase_success_201(self, mock_post):
        """Verify successful HTTP 201 response returns (True, None)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_post.return_value = mock_resp

        success, err = submit_feedback_to_supabase(
            url="https://testproject.supabase.co/",
            publishable_key="test_anon_key",
            rating=5,
            suggestions="Excellent analytics studio!",
        )

        self.assertTrue(success)
        self.assertIsNone(err)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://testproject.supabase.co/rest/v1/feedback")
        self.assertEqual(kwargs["headers"]["apikey"], "test_anon_key")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_anon_key")
        self.assertEqual(kwargs["headers"]["Prefer"], "return=minimal")
        self.assertEqual(kwargs["json"], {"rating": 5, "suggestions": "Excellent analytics studio!"})

    @patch("requests.post")
    def test_submit_feedback_to_supabase_optional_suggestions_none(self, mock_post):
        """Verify suggestions can be None when user only submits a star rating."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        success, err = submit_feedback_to_supabase(
            url="https://testproject.supabase.co",
            publishable_key="test_anon_key",
            rating=4,
            suggestions=None,
        )

        self.assertTrue(success)
        self.assertIsNone(err)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"], {"rating": 4, "suggestions": None})

    @patch("requests.post")
    def test_submit_feedback_to_supabase_http_error(self, mock_post):
        """Verify HTTP 403 Forbidden (RLS violation) returns (False, error_msg)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"message": "new row violates row-level security policy"}
        mock_post.return_value = mock_resp

        success, err = submit_feedback_to_supabase(
            url="https://testproject.supabase.co",
            publishable_key="test_anon_key",
            rating=3,
        )

        self.assertFalse(success)
        self.assertIn("Supabase API returned 403", err)
        self.assertIn("violates row-level security policy", err)

    @patch("requests.post")
    def test_submit_feedback_to_supabase_timeout_handling(self, mock_post):
        """Verify network timeout is trapped and returned as a clean error message."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        success, err = submit_feedback_to_supabase(
            url="https://testproject.supabase.co",
            publishable_key="test_anon_key",
            rating=5,
        )

        self.assertFalse(success)
        self.assertIn("timed out", err.lower())

    @patch("requests.post")
    def test_submit_feedback_to_supabase_connection_error(self, mock_post):
        """Verify connection error is trapped and returned as a clean error message."""
        mock_post.side_effect = requests.exceptions.ConnectionError("DNS lookup failed")

        success, err = submit_feedback_to_supabase(
            url="https://testproject.supabase.co",
            publishable_key="test_anon_key",
            rating=5,
        )

        self.assertFalse(success)
        self.assertIn("unable to establish network connection", err.lower())

    # -------------------------------------------------------------------------
    # 3. UI Component State & Feedback Workflow Tests
    # -------------------------------------------------------------------------
    def test_render_feedback_section_unsubmitted_clean(self):
        """Verify feedback section renders cleanly in unsubmitted state without crashing."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("streamlit.feedback", return_value=None), \
             patch("streamlit.text_area", return_value=""), \
             patch("streamlit.button", return_value=False):
            render_feedback_section()
            self.assertFalse(st.session_state["feedback_submitted"])

    def test_render_feedback_section_submitted_success_state(self):
        """Verify feedback section renders success message when feedback_submitted is True."""
        state_submitted = {"feedback_submitted": True}
        with patch.dict(st.session_state, state_submitted, clear=True), \
             patch("streamlit.success") as mock_success, \
             patch("streamlit.button", return_value=False):
            render_feedback_section()
            mock_success.assert_called_once()
            self.assertTrue(st.session_state["feedback_submitted"])

    def test_render_feedback_warns_when_star_not_selected(self):
        """Verify warning is shown if Submit is clicked without selecting stars."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("streamlit.feedback", return_value=None), \
             patch("streamlit.text_area", return_value="Great app!"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.warning") as mock_warning, \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            mock_warning.assert_called_once()
            self.assertFalse(st.session_state["feedback_submitted"])
            mock_rerun.assert_not_called()

    def test_render_feedback_warns_when_supabase_credentials_missing_locally(self):
        """Verify clear configuration warning is shown locally when secrets are absent."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("src.ui.feedback.get_supabase_credentials", return_value=(None, None)), \
             patch("streamlit.feedback", return_value=4), \
             patch("streamlit.text_area", return_value="Feedback"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.warning") as mock_warning, \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            mock_warning.assert_called_once()
            self.assertIn("Supabase credentials not configured", mock_warning.call_args[0][0])
            self.assertFalse(st.session_state["feedback_submitted"])
            mock_rerun.assert_not_called()

    @patch("src.ui.feedback.submit_feedback_to_supabase")
    def test_render_feedback_successful_submission_triggers_rerun(self, mock_submit):
        """Verify successful Supabase submission sets session state and reruns."""
        mock_submit.return_value = (True, None)

        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("src.ui.feedback.get_supabase_credentials", return_value=("https://sb.co", "key123")), \
             patch("streamlit.feedback", return_value=4), \
             patch("streamlit.text_area", return_value="Love the UI!"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            mock_submit.assert_called_once_with(
                url="https://sb.co",
                publishable_key="key123",
                rating=5,  # star index 4 + 1 = 5 stars
                suggestions="Love the UI!",
            )
            self.assertTrue(st.session_state["feedback_submitted"])
            mock_rerun.assert_called_once()

    @patch("src.ui.feedback.submit_feedback_to_supabase")
    def test_render_feedback_database_failure_shows_error_no_rerun(self, mock_submit):
        """Verify database error displays error message without showing success or crashing."""
        mock_submit.return_value = (False, "Supabase API returned 500: Internal server error")

        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("src.ui.feedback.get_supabase_credentials", return_value=("https://sb.co", "key123")), \
             patch("streamlit.feedback", return_value=3), \
             patch("streamlit.text_area", return_value="Some bug"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.error") as mock_error, \
             patch("streamlit.rerun") as mock_rerun:
            render_feedback_section()

            mock_submit.assert_called_once()
            mock_error.assert_called_once()
            self.assertIn("Failed to submit feedback to database", mock_error.call_args[0][0])
            self.assertFalse(st.session_state["feedback_submitted"])
            mock_rerun.assert_not_called()

    def test_overview_view_integration(self):
        """Verify Overview page renders cleanly with feedback section integrated."""
        with patch.dict(st.session_state, self.mock_state, clear=True), \
             patch("src.ui.views.overview.render_feedback_section") as mock_fb_render:
            render_overview_view()
            mock_fb_render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
