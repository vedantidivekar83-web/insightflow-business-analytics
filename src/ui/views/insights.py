"""Smart Insights dashboard view for InsightFlow.

Renders contextual, data-backed observations across Data Quality, Business,
Numerical anomalies, Categorical distributions, Correlations, and Trends
using native Streamlit components.
"""

from typing import Any
import streamlit as st
import pandas as pd

from src.analytics.insights import generate_smart_insights


SEVERITY_META: dict[str, dict[str, str]] = {
    "Critical": {
        "icon": "🚨",
        "badge": ":red[● Critical]",
        "pill": "<span style='background: rgba(251, 113, 133, 0.15); color: #FB7185; border: 1px solid rgba(251, 113, 133, 0.4); padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.74rem; font-weight: 600;'>● Critical</span>",
    },
    "Warning": {
        "icon": "⚠️",
        "badge": ":orange[● Warning]",
        "pill": "<span style='background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.4); padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.74rem; font-weight: 600;'>● Warning</span>",
    },
    "Info": {
        "icon": "💡",
        "badge": ":blue[● Info]",
        "pill": "<span style='background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.4); padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.74rem; font-weight: 600;'>● Info</span>",
    },
    "Positive": {
        "icon": "✅",
        "badge": ":green[● Positive]",
        "pill": "<span style='background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.74rem; font-weight: 600;'>● Positive</span>",
    },
}


def render_insights_view() -> None:
    """Render the Smart Insights interface."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # 1. Empty State Guard
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Smart Insights Engine
                </h2>
                <p class="if-page-subtitle">
                    Automated, data-driven observations, anomaly highlights, and strategic takeaways.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">💡</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset for Smart Insights
                </h3>
                <p style="margin: 0 auto; max-width: 500px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Please navigate to <strong>Data Upload</strong> in the sidebar to upload a CSV or Excel dataset first.
                    Once active, InsightFlow will automatically evaluate data quality,
                    statistical outliers, categorical imbalances, and time trends.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filename = metadata["filename"] if metadata else "Active Dataset"

    # Header section
    st.markdown(
        f"""
        <div class="if-page-header">
            <h2 class="if-page-title">
                Smart Insights Engine
            </h2>
            <p class="if-page-subtitle">
                Active Dataset: <strong style="color: #22D3EE;">{filename}</strong>
                &bull; {len(active_df):,} records analyzed
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Generate Insights
    with st.spinner("Analyzing dataset patterns and synthesizing observations..."):
        insights = generate_smart_insights(active_df)

    if not insights:
        with st.container(border=True):
            st.markdown(
                """### 🔍 No Specific Anomaly or Severe Pattern Detected
The active dataset exhibits standard uniform distributions with no critical outliers,
excessive missingness, or severe categorical imbalance.
Review the **Analytics** section for detailed statistical summaries and distribution charts.
"""
            )
        return

    # 3. Summary Scorecard Row
    total_count = len(insights)
    critical_count = sum(1 for i in insights if i.get("severity") == "Critical")
    warning_count = sum(1 for i in insights if i.get("severity") == "Warning")
    positive_count = sum(1 for i in insights if i.get("severity") == "Positive")
    info_count = sum(1 for i in insights if i.get("severity") == "Info")

    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    with sc_col1:
        st.metric(label="Total Insights", value=total_count)
    with sc_col2:
        st.metric(label="Attention Required", value=critical_count + warning_count)
    with sc_col3:
        st.metric(label="Operational Strengths", value=positive_count)
    with sc_col4:
        st.metric(label="Informational Patterns", value=info_count)

    st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)

    # 4. Category Filter Selector
    available_categories = ["All Categories"] + sorted(list({str(i["category"]) for i in insights}))
    col_filter, _ = st.columns([2, 2])
    with col_filter:
        selected_category = st.selectbox(
            label="Filter Insights by Category",
            options=available_categories,
            index=0,
            key="insights_cat_filter",
        )

    # Filtered insights list
    filtered_insights = (
        insights
        if selected_category == "All Categories"
        else [i for i in insights if i.get("category") == selected_category]
    )

    st.caption(f"Showing **{len(filtered_insights)}** of **{total_count}** insights")

    # 5. Render Individual Insight Cards Using Native Streamlit Components
    for item in filtered_insights:
        sev = item.get("severity", "Info")
        meta = SEVERITY_META.get(sev, SEVERITY_META["Info"])

        with st.container(border=True):
            header_col, meta_col = st.columns([3, 1])
            with header_col:
                st.markdown(f"#### {meta['icon']} {item['title']}")
            with meta_col:
                st.markdown(f"<div style='text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem;'>{meta['pill']} <span class='if-badge if-badge-neutral' style='font-size: 0.76rem;'>{item['category']}</span></div>", unsafe_allow_html=True)

            # Clean markdown description (no raw HTML tags exposed)
            st.markdown(item["description"])

            # Numerical Evidence Badges / Chips
            evidence_dict = item.get("evidence", {})
            if evidence_dict:
                evidence_parts = [f"**{k}:** `{v}`" for k, v in evidence_dict.items()]
                st.markdown(" &nbsp;•&nbsp; ".join(evidence_parts))

            # Contextual "Why this matters" callout
            st.markdown(f"> 💡 **Why this matters:** {item['why_it_matters']}")
