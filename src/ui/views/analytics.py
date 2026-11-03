"""Business Analytics view for InsightFlow.

Renders statistical summaries, data quality audits, business KPIs,
correlation heatmaps, and distribution charts using Plotly.
"""

from typing import Any
import streamlit as st
import pandas as pd
import numpy as np

from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import (
    compute_data_quality_summary,
    compute_column_quality_table,
)
from src.analytics.kpi import compute_business_kpis
from src.analytics.correlation import (
    compute_correlation_matrix,
    get_top_correlations,
)
from src.visualization.charts import (
    plot_numerical_distribution,
    plot_categorical_frequency,
    plot_correlation_heatmap,
    plot_time_series_trend,
)


def render_analytics_view() -> None:
    """Render the Business Analytics Engine interface."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # 1. Empty State Guard
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Business Analytics & Decision Intelligence
                </h2>
                <p class="if-page-subtitle">
                    Automated statistical summaries, data quality scoring, and correlation discovery.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📈</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset for Analytics
                </h3>
                <p style="margin: 0 auto; max-width: 500px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Upload a CSV or Excel dataset in <strong>Data Upload</strong> to compute statistical metrics,
                    business KPIs, data quality audits, and interactive correlation matrices.
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
                Business Analytics & Decision Intelligence
            </h2>
            <p class="if-page-subtitle">
                Active Dataset: <strong style="color: #22D3EE;">{filename}</strong>
                &bull; {len(active_df):,} records &bull; {len(active_df.columns)} dimensions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Analytics Tabs
    tab_kpi, tab_stats, tab_quality, tab_visuals = st.tabs(
        [
            "📊  Business KPIs",
            "📐  Statistical Summaries",
            "🛡️  Data Quality Audit",
            "📈  Visual Analytics",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: Business KPIs & Inferred Metrics
    # ---------------------------------------------------------
    with tab_kpi:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Executive Business KPIs</h3>",
            unsafe_allow_html=True,
        )

        kpi_results = compute_business_kpis(active_df)
        inferred = kpi_results["inferred_kpis"]

        if inferred:
            # Render inferred revenue / transaction cards
            kpi_cols = st.columns(len(inferred))
            for i, kpi in enumerate(inferred):
                with kpi_cols[i]:
                    st.metric(label=kpi["name"], value=kpi["formatted"])
                    st.caption(f"💡 *{kpi['assumption']}*")
            st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)
        else:
            # Non-sales dataset notice & primary metric fallback
            st.info(
                "ℹ️ **General Business Dataset**: No recognizable sales/revenue columns were detected. "
                "Displaying dimensional volume and primary numerical metric metrics below."
            )

        # Primary Numerical Metric Summary (if present)
        primary = kpi_results["primary_metric"]
        if primary:
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric(label=f"Total {primary['column_name']}", value=f"{primary['sum']:,.2f}")
            with p_col2:
                st.metric(label=f"Average {primary['column_name']}", value=f"{primary['mean']:,.2f}")
            with p_col3:
                st.metric(label=f"Minimum {primary['column_name']}", value=f"{primary['min']:,.2f}")
            with p_col4:
                st.metric(label=f"Maximum {primary['column_name']}", value=f"{primary['max']:,.2f}")
            st.caption(f"💡 *{primary['assumption']}*")

        # Status Distribution (if detected)
        status_df = kpi_results["status_distribution"]
        if status_df is not None and not status_df.empty:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h4 style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;'>Status Distribution Breakdown</h4>",
                unsafe_allow_html=True,
            )
            col_tbl, col_chart = st.columns([1, 1])
            with col_tbl:
                st.dataframe(status_df, use_container_width=True, hide_index=True)
            with col_chart:
                status_col_name = kpi_results["detected_columns"]["status"]
                if status_col_name:
                    fig_status = plot_categorical_frequency(active_df, status_col_name, top_n=8)
                    st.plotly_chart(fig_status, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: Statistical Summaries
    # ---------------------------------------------------------
    with tab_stats:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Descriptive Statistics (Numerical Columns)</h3>",
            unsafe_allow_html=True,
        )

        stats_df = compute_numerical_statistics(active_df)
        if stats_df.empty:
            st.warning("⚠️ No numerical columns found in the active dataset to compute descriptive statistics.")
        else:
            st.dataframe(stats_df, use_container_width=True)
            st.caption("Includes central tendency (Mean, Median, Mode), dispersion (Std Dev, Variance), and range quartiles (Q1, Q2, Q3, IQR).")

    # ---------------------------------------------------------
    # TAB 3: Data Quality Audit
    # ---------------------------------------------------------
    with tab_quality:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Dataset Integrity & Completeness Audit</h3>",
            unsafe_allow_html=True,
        )

        quality_summary = compute_data_quality_summary(active_df)

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric(label="Overall Completeness", value=f"{quality_summary['completeness_pct']}%")
        with q_col2:
            st.metric(label="Missing Cells", value=f"{quality_summary['missing_cells']:,} ({quality_summary['missing_pct']}%)")
        with q_col3:
            st.metric(label="Duplicate Rows", value=f"{quality_summary['duplicate_rows']:,} ({quality_summary['duplicate_pct']}%)")
        with q_col4:
            st.metric(label="Total Data Points", value=f"{quality_summary['total_cells']:,}")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;'>Column-by-Column Quality Breakdown</h4>",
            unsafe_allow_html=True,
        )

        col_quality_df = compute_column_quality_table(active_df)
        st.dataframe(col_quality_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 4: Visual Analytics
    # ---------------------------------------------------------
    with tab_visuals:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Interactive Visual Analytics</h3>",
            unsafe_allow_html=True,
        )

        num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in active_df.columns if c not in num_cols]
        date_candidates = [
            c for c in active_df.columns
            if pd.api.types.is_datetime64_any_dtype(active_df[c]) or "date" in str(c).lower() or "time" in str(c).lower()
        ]

        # Section 1: Distribution Analysis & Categorical Frequency
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            if num_cols:
                selected_num = st.selectbox("Select Numerical Feature for Distribution", options=num_cols, key="vis_num")
                fig_dist = plot_numerical_distribution(active_df, selected_num)
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No numerical features available for distribution histogram.")

        with vis_col2:
            if cat_cols:
                selected_cat = st.selectbox("Select Categorical Feature for Frequency", options=cat_cols, key="vis_cat")
                fig_cat = plot_categorical_frequency(active_df, selected_cat, top_n=10)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical features available for frequency chart.")

        st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)

        # Section 2: Correlation Analysis
        st.markdown(
            "<h4 style='font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;'>Feature Correlation Analysis</h4>",
            unsafe_allow_html=True,
        )

        corr_matrix, dropped_constants = compute_correlation_matrix(active_df)
        if corr_matrix.empty:
            st.info("ℹ️ Correlation requires at least 2 non-constant numerical columns.")
        else:
            if dropped_constants:
                st.caption(f"Note: Excluded constant columns with zero variance: {', '.join(dropped_constants)}")

            corr_col, table_col = st.columns([3, 2])
            with corr_col:
                fig_corr = plot_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)

            with table_col:
                st.markdown(
                    "<p style='font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-top: 0.5rem;'>Top Pairwise Correlations</p>",
                    unsafe_allow_html=True,
                )
                top_pairs = get_top_correlations(corr_matrix, top_n=8)
                if top_pairs:
                    top_df = pd.DataFrame(top_pairs)[["feature_a", "feature_b", "correlation", "relationship"]]
                    top_df.columns = ["Feature A", "Feature B", "Pearson r", "Relationship"]
                    st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Section 3: Time-Series Trends (if dates exist)
        if date_candidates and num_cols:
            st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h4 style='font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem;'>Time-Series Trend Analysis</h4>",
                unsafe_allow_html=True,
            )
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                selected_date = st.selectbox("Date / Timestamp Column", options=date_candidates, key="trend_date")
            with t_col2:
                selected_metric = st.selectbox("Metric Column", options=num_cols, key="trend_metric")

            fig_trend = plot_time_series_trend(active_df, selected_date, selected_metric)
            st.plotly_chart(fig_trend, use_container_width=True)
