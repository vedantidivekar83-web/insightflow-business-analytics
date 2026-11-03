"""InsightFlow - Smart Business Data Analytics & Decision Support System.

Main application entry point orchestrating modular UI presentation layers.
"""

import streamlit as st

from src.ui.styles import apply_custom_styles
from src.ui.header import render_header
from src.ui.sidebar import render_sidebar
from src.ui.views.overview import render_overview_view
from src.ui.views.upload import render_upload_view
from src.ui.views.explorer import render_explorer_view
from src.ui.views.analytics import render_analytics_view
from src.ui.views.advanced_analytics import render_advanced_analytics_view
from src.ui.views.ml_insights import render_ml_insights_view
from src.ui.views.insights import render_insights_view
from src.ui.views.reports import render_reports_view
from src.ui.views.placeholder import render_placeholder_view


def main() -> None:
    """Initialize and run the InsightFlow application."""
    st.set_page_config(
        page_title="InsightFlow | Turn Data Into Decisions",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply modern SaaS styling
    apply_custom_styles()

    # Render sidebar navigation
    selected_page = render_sidebar()

    # Render top branding header with active page context
    render_header(selected_page)

    # View Routing
    if selected_page == "Overview":
        render_overview_view()
    elif selected_page == "Data Upload":
        render_upload_view()
    elif selected_page == "Data Explorer":
        render_explorer_view()
    elif selected_page == "Analytics":
        render_analytics_view()
    elif selected_page == "Advanced Analytics":
        render_advanced_analytics_view()
    elif selected_page == "Machine Learning":
        render_ml_insights_view()
    elif selected_page == "Smart Insights":
        render_insights_view()
    elif selected_page == "Reports":
        render_reports_view()
    else:
        render_placeholder_view(selected_page)


if __name__ == "__main__":
    main()
