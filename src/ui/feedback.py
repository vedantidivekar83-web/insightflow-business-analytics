"""Compact Feedback and Suggestions component with Supabase integration.

Allows users to submit a 1–5 star experience rating and optional suggestions
directly from the Overview page into a Supabase 'feedback' table using an
anonymous insert policy without collecting personal identifying information (PII)
or uploaded dataset contents.
"""

from typing import Any
import streamlit as st
import requests


def get_supabase_credentials() -> tuple[str | None, str | None]:
    """Retrieve Supabase URL and Publishable Key safely from Streamlit secrets.

    Returns
    -------
    tuple[str | None, str | None]
        (url, publishable_key) if available in st.secrets["supabase"], else (None, None).
    """
    try:
        if "supabase" in st.secrets:
            supabase_conf = st.secrets["supabase"]
            url = supabase_conf.get("url")
            key = supabase_conf.get("publishable_key")
            if url and key:
                return str(url).strip(), str(key).strip()
    except Exception:
        # st.secrets might be unavailable, missing, or raise PermissionError in local setups
        pass
    return None, None


def submit_feedback_to_supabase(
    url: str,
    publishable_key: str,
    rating: int,
    suggestions: str | None = None,
    timeout: float = 8.0,
) -> tuple[bool, str | None]:
    """Insert a feedback entry into the Supabase 'feedback' table.

    Parameters
    ----------
    url : str
        The Supabase project URL (e.g. 'https://xyz.supabase.co').
    publishable_key : str
        The Supabase publishable / anon key.
    rating : int
        User experience rating (1 to 5).
    suggestions : str | None, optional
        Optional suggestion text.
    timeout : float, optional
        Network timeout in seconds (default: 8.0).

    Returns
    -------
    tuple[bool, str | None]
        (True, None) on successful insertion.
        (False, error_message) if submission or connection failed.
    """
    clean_url = url.strip().rstrip("/")
    endpoint = f"{clean_url}/rest/v1/feedback"

    headers = {
        "apikey": publishable_key.strip(),
        "Authorization": f"Bearer {publishable_key.strip()}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    payload: dict[str, Any] = {
        "rating": int(rating),
        "suggestions": suggestions.strip() if suggestions else None,
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        # PostgREST returns HTTP 201 Created on successful insert
        if response.status_code in (200, 201):
            return True, None

        # Parse error message from response if available
        try:
            err_data = response.json()
            err_msg = err_data.get("message") or err_data.get("error") or response.text
        except Exception:
            err_msg = response.text or f"HTTP {response.status_code}"

        return False, f"Supabase API returned {response.status_code}: {err_msg}"

    except requests.exceptions.Timeout:
        return False, "Request timed out while connecting to Supabase database."
    except requests.exceptions.ConnectionError:
        return False, "Unable to establish network connection to Supabase."
    except Exception as exc:
        return False, f"Unexpected error during database submission: {str(exc)}"


def render_feedback_section() -> None:
    """Render a compact, theme-aligned feedback & suggestions widget inside Overview."""
    # Ensure session state variables exist
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = False

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        # Header Row
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    <span style="font-size: 1.25rem;">💬</span>
                    <div>
                        <h3 style="margin: 0; font-size: 1.08rem; color: #F8FAFC; font-weight: 700; letter-spacing: -0.01em;">
                            Feedback &amp; Suggestions
                        </h3>
                        <p style="margin: 0.15rem 0 0 0; font-size: 0.82rem; color: #A5A5C5;">
                            Help refine InsightFlow &bull; Rate your experience and share thoughts
                        </p>
                    </div>
                </div>
                <span class="if-badge if-badge-cyan">Anonymous &bull; 1–5 ★</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state["feedback_submitted"]:
            st.success("🎉 **Thank you!** Your feedback has been securely submitted to the database.")
            if st.button("✏️  Submit Another Response", key="btn_reset_feedback"):
                st.session_state["feedback_submitted"] = False
                st.rerun()
            return

        col_rating, col_desc = st.columns([1, 2])
        with col_rating:
            st.markdown(
                "<label style='font-size: 0.86rem; color: #F8FAFC; font-weight: 600; display: block; margin-bottom: 0.25rem;'>"
                "Experience Rating <span style='color: #EC4899;'>*</span>"
                "</label>",
                unsafe_allow_html=True,
            )
            if hasattr(st, "feedback"):
                star_idx = st.feedback("stars", key="overview_feedback_stars")
                rating = (star_idx + 1) if star_idx is not None else None
            else:
                rating = st.select_slider(
                    "Rating",
                    options=[1, 2, 3, 4, 5],
                    value=5,
                    format_func=lambda x: f"{'⭐' * x} ({x})",
                    key="overview_feedback_stars_fallback",
                    label_visibility="collapsed",
                )

        with col_desc:
            rating_labels = {
                1: "⭐ Needs Work",
                2: "⭐⭐ Fair",
                3: "⭐⭐⭐ Good",
                4: "⭐⭐⭐⭐ Very Good",
                5: "⭐⭐⭐⭐⭐ Outstanding!",
            }
            if rating is not None:
                rating_badge = f"<span class='if-badge if-badge-pink' style='margin-top: 1.85rem; display: inline-block;'>{rating_labels.get(rating, f'{rating} Stars')}</span>"
            else:
                rating_badge = "<span style='color: #71719A; font-size: 0.82rem; margin-top: 2rem; display: inline-block;'>Click a star to rate (1–5)</span>"
            st.markdown(rating_badge, unsafe_allow_html=True)

        suggestions = st.text_area(
            label="Suggestions & Enhancements (Optional)",
            placeholder="Share any ideas, missing analytics features, or feedback to improve InsightFlow... (No personal details or dataset contents are collected)",
            max_chars=500,
            height=85,
            key="overview_feedback_suggestions",
        )

        submit_btn = st.button(
            "✨  Submit Feedback",
            type="primary",
            use_container_width=True,
            key="btn_submit_feedback",
        )

        if submit_btn:
            if rating is None:
                st.warning("⚠️ Please select a 1–5 star rating above before submitting.")
                return

            url, pub_key = get_supabase_credentials()
            if not url or not pub_key:
                st.warning(
                    "⚠️ **Supabase credentials not configured.** "
                    "To persist feedback to the database, configure `[supabase]` with `url` and `publishable_key` in `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
                )
                return

            with st.spinner("Submitting feedback to database..."):
                success, err_msg = submit_feedback_to_supabase(
                    url=url,
                    publishable_key=pub_key,
                    rating=int(rating),
                    suggestions=suggestions.strip() if suggestions else None,
                )

            if success:
                st.session_state["feedback_submitted"] = True
                st.rerun()
            else:
                st.error(f"❌ **Failed to submit feedback to database:** {err_msg}")
