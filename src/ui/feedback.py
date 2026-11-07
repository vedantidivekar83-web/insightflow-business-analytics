"""Compact Feedback and Suggestions component for InsightFlow.

Allows users to submit a 1–5 star experience rating and optional suggestions
directly from the Overview page without collecting personal identifying
information (PII) or uploaded dataset contents.
"""

from typing import Any
import datetime
import json
import os
import streamlit as st


def save_feedback_entry(entry: dict[str, Any]) -> None:
    """Record user feedback into session state and attempt local append.

    Parameters
    ----------
    entry : dict[str, Any]
        Dictionary containing 'timestamp', 'rating' (1-5), and optional 'suggestions'.
    """
    if "feedback_history" not in st.session_state:
        st.session_state["feedback_history"] = []
    st.session_state["feedback_history"].append(entry)

    # Optional local file audit trail (for local runs)
    # Note: On Streamlit Community Cloud, container storage is ephemeral
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "feedback.jsonl")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Gracefully ignore filesystem write errors (e.g., read-only cloud filesystems)
        pass


def render_feedback_section() -> None:
    """Render a compact, theme-aligned feedback & suggestions widget inside Overview."""
    # Ensure session state variables exist
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = False
    if "feedback_history" not in st.session_state:
        st.session_state["feedback_history"] = []

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
            st.success("🎉 **Thank you!** Your feedback and suggestions have been recorded for this session.")
            st.markdown(
                """
                <p style="font-size: 0.78rem; color: #71719A; margin-top: -0.25rem; margin-bottom: 0.75rem; line-height: 1.45;">
                    <em>Storage note:</em> Feedback is retained in your active browser session. On Streamlit Community Cloud's ephemeral container architecture, cross-session persistence across cloud reboots requires an external cloud database or service.
                </p>
                """,
                unsafe_allow_html=True,
            )
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
            else:
                entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "rating": int(rating),
                    "suggestions": suggestions.strip() if suggestions else None,
                }
                save_feedback_entry(entry)
                st.session_state["feedback_submitted"] = True
                st.rerun()
