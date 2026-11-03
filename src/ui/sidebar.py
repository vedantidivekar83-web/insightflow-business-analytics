"""Sidebar navigation and status indicator component.

Provides modern Futuristic Purple Glass Studio navigation with glowing pills,
active indicator states, and real-time system status synchronization.
"""

from typing import Literal
import streamlit as st

NavSection = Literal[
    "Overview",
    "Data Upload",
    "Data Explorer",
    "Analytics",
    "Advanced Analytics",
    "Machine Learning",
    "Smart Insights",
    "Reports",
]

NAV_MAP: dict[str, NavSection] = {
    "⚡  Overview": "Overview",
    "🏠  Overview": "Overview",
    "📤  Data Upload": "Data Upload",
    "🔍  Data Explorer": "Data Explorer",
    "📈  Analytics": "Analytics",
    "🔬  Advanced Analytics": "Advanced Analytics",
    "🤖  Machine Learning": "Machine Learning",
    "💡  Smart Insights": "Smart Insights",
    "📋  Reports": "Reports",
}

NAV_DISPLAY_OPTIONS: list[str] = [
    "⚡  Overview",
    "📤  Data Upload",
    "🔍  Data Explorer",
    "📈  Analytics",
    "🔬  Advanced Analytics",
    "🤖  Machine Learning",
    "💡  Smart Insights",
    "📋  Reports",
]


def render_sidebar() -> NavSection:
    """Render the sidebar navigation and return the currently selected section."""
    with st.sidebar:
        # App Branding Header
        st.markdown(
            """
            <div style="padding: 0.35rem 0 1.25rem 0; display: flex; align-items: center; gap: 0.75rem;">
                <div style="
                    width: 36px;
                    height: 36px;
                    border-radius: 11px;
                    background: linear-gradient(135deg, #8B5CF6 0%, #A855F7 50%, #EC4899 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #FFFFFF;
                    font-size: 1.2rem;
                    box-shadow: 0 4px 16px rgba(168, 85, 247, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                ">
                    ⚡
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.28rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.025em; line-height: 1.2;">
                        InsightFlow
                    </h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #A5A5C5; font-weight: 500;">
                        Purple Glass Studio
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #7C7C9E; letter-spacing: 0.1em; margin: 0.2rem 0 0.5rem 0.25rem;'>Studio Workspaces</p>",
            unsafe_allow_html=True,
        )

        # Radio Navigation Menu (Pill-styled via styles.py)
        selected_display = st.radio(
            label="Navigation Menu",
            options=NAV_DISPLAY_OPTIONS,
            label_visibility="collapsed",
            index=0,
        )

        st.markdown("<hr style='margin: 1.25rem 0 1rem 0; border: none; border-top: 1px solid rgba(167, 139, 250, 0.18);'/>", unsafe_allow_html=True)

        # System Status Summary Card
        st.markdown(
            "<p style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #7C7C9E; letter-spacing: 0.1em; margin: 0 0 0.5rem 0.25rem;'>Workspace Engine</p>",
            unsafe_allow_html=True,
        )

        metadata = st.session_state.get("dataset_metadata")
        if metadata and metadata.get("filename"):
            full_name = str(metadata["filename"])
            if len(full_name) > 20:
                shortened_name = full_name[:17] + "..."
            else:
                shortened_name = full_name
            rows_count = f"{metadata.get('total_rows', 0):,}"
            dataset_status_html = (
                f"<span style='color: #22D3EE; font-weight: 600;' title='{full_name}'>{shortened_name}</span>"
                f"<div style='font-size: 0.76rem; color: #A5A5C5;'>({rows_count} rows)</div>"
            )
            engine_status_label = "<span style='color: #34D399; font-weight: 600; font-size: 0.75rem;'>● Active</span>"
        else:
            dataset_status_html = "<span style='color: #7C7C9E;'>None loaded</span>"
            engine_status_label = "<span style='color: #7C7C9E; font-weight: 600; font-size: 0.75rem;'>○ Idle</span>"

        st.markdown(
            f"""
            <div class="if-status-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem; padding-bottom: 0.45rem; border-bottom: 1px solid rgba(167, 139, 250, 0.15);">
                    <span style="font-size: 0.75rem; color: #A5A5C5;">Engine Status</span>
                    {engine_status_label}
                </div>
                <div style="margin-bottom: 0.35rem;"><strong>Environment:</strong> Python 3.11</div>
                <div><strong>Active Dataset:</strong> {dataset_status_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return NAV_MAP.get(selected_display, "Overview")
