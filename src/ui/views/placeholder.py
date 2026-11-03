"""Placeholder views for scheduled modules.

Provides clean empty-state guidance without dummy data or premature features.
"""

import streamlit as st

SECTION_METADATA: dict[str, dict[str, str]] = {}


def render_placeholder_view(section_name: str) -> None:
    """Render a clean architectural placeholder for an unactivated module."""
    meta = SECTION_METADATA.get(
        section_name,
        {
            "title": section_name,
            "description": "This module is slated for implementation in an upcoming development stage.",
            "stage": "Scheduled Module",
            "icon": "⚙️",
        },
    )

    st.markdown(
        f"""
        <div class="if-placeholder-box">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{meta['icon']}</div>
            <span class="if-badge" style="margin-bottom: 0.75rem;">{meta['stage']}</span>
            <h3 style="margin: 0.5rem 0 0.35rem 0; font-size: 1.35rem; color: #0F172A; font-weight: 600;">
                {meta['title']}
            </h3>
            <p style="margin: 0 auto; max-width: 600px; font-size: 0.92rem; color: #64748B; line-height: 1.5;">
                {meta['description']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
