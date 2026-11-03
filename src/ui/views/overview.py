"""Overview landing page view for InsightFlow.

Enforces the 'Futuristic Purple Glass Analytics Studio' visual design system.
Features a multi-tiered executive layout:
- Top welcome header & active dataset status banner
- Executive 4-metric KPI row (Total Records, Dimensions, Completeness, Numerical Features)
- Responsive visualization grid (live Plotly distribution & categorical charts when dataset active)
- Studio capabilities & three-step intelligence workflow roadmap
"""

from typing import Any
import streamlit as st
import pandas as pd

from src.data.explorer import classify_columns
from src.analytics.quality import compute_data_quality_summary
from src.visualization.charts import (
    plot_numerical_distribution,
    plot_categorical_frequency,
)


def render_overview_view() -> None:
    """Render the main Overview analytics studio landing view."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # =========================================================================
    # 1. TOP SECTION: Hero Banner & Workspace Context
    # =========================================================================
    st.markdown(
        """
        <div class="if-hero-banner">
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span class="if-badge">⚡ Purple Glass Studio</span>
                <span class="if-badge if-badge-cyan">Autonomous Decision Intelligence</span>
            </div>
            <h2 class="if-hero-title">
                Executive Business Analytics &amp; Machine Learning Studio
            </h2>
            <p class="if-hero-desc">
                InsightFlow transforms operational datasets into strategic intelligence.
                Harness automated exploratory profiling, statistical KPI synthesis,
                unsupervised anomaly detection, and forward-looking time-series forecasting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 2. DATASET STATUS & PRIMARY ACTION
    # =========================================================================
    if active_df is not None and metadata is not None and not active_df.empty:
        status_col, action_col = st.columns([3, 1])
        with status_col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(90deg, rgba(34, 211, 238, 0.14) 0%, rgba(139, 92, 246, 0.14) 100%);
                    border: 1px solid rgba(167, 139, 250, 0.35);
                    border-radius: 14px;
                    padding: 0.9rem 1.35rem;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                ">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.25rem;">⚡</span>
                        <div>
                            <span style="color: #F8FAFC; font-size: 0.94rem; font-weight: 600;">
                                Active Dataset: <strong style="color: #22D3EE;">{metadata.get('filename')}</strong>
                            </span>
                            <span style="color: #A5A5C5; font-size: 0.85rem; margin-left: 0.75rem;">
                                ({len(active_df):,} rows &bull; {len(active_df.columns)} dimensions &bull; {metadata.get('file_size_formatted', 'N/A')})
                            </span>
                        </div>
                    </div>
                    <span class="if-badge if-badge-cyan">Workspace Synchronized</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_col:
            st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
            if st.button("📁  Change Dataset", use_container_width=True):
                # Inform user to navigate to Upload
                st.session_state["_nav_to_upload"] = True
                st.info("Navigate to **📤 Data Upload** in the sidebar to load a new file.")
    else:
        st.markdown(
            """
            <div style="
                background: rgba(23, 23, 47, 0.75);
                border: 1px solid rgba(167, 139, 250, 0.2);
                border-radius: 14px;
                padding: 0.85rem 1.35rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.25rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    <span style="font-size: 1.15rem;">📁</span>
                    <span style="color: #A5A5C5; font-size: 0.9rem;">
                        Workspace Standby &bull; No dataset currently loaded. Connect a CSV or Excel file to unlock live visual analysis.
                    </span>
                </div>
                <span class="if-badge if-badge-neutral">Awaiting Ingestion</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. EXECUTIVE KPI ROW
    # =========================================================================
    if active_df is not None and not active_df.empty:
        # Compute real dataset metrics
        classified = classify_columns(active_df)
        num_cols = classified["numerical"]
        quality_summary = compute_data_quality_summary(active_df)

        total_records_val = f"{len(active_df):,}"
        total_dims_val = f"{len(active_df.columns)}"
        completeness_val = f"{quality_summary.get('completeness_pct', 100.0)}%"
        num_cols_val = f"{len(num_cols)}"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(label="Total Records", value=total_records_val)
        with kpi2:
            st.metric(label="Total Dimensions", value=total_dims_val)
        with kpi3:
            st.metric(label="Data Completeness", value=completeness_val)
        with kpi4:
            st.metric(label="Numerical Features", value=num_cols_val)
    else:
        # Meaningful onboarding metric cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(label="Total Records", value="0", help="Awaiting CSV/Excel dataset upload")
        with kpi2:
            st.metric(label="Total Dimensions", value="0", help="Schema detection will profile columns upon upload")
        with kpi3:
            st.metric(label="Data Completeness", value="—", help="Data quality audit ready")
        with kpi4:
            st.metric(label="Numerical Features", value="0", help="Statistical engine on standby")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. MAIN VISUALIZATION GRID
    # =========================================================================
    if active_df is not None and not active_df.empty:
        st.markdown(
            "<h3 style='font-size: 1.22rem; font-weight: 700; color: #F8FAFC; margin: 0 0 1rem 0; letter-spacing: -0.02em;'>Live Intelligence Overview</h3>",
            unsafe_allow_html=True,
        )

        classified = classify_columns(active_df)
        num_cols = classified["numerical"]
        cat_cols = classified["categorical"]

        chart_left, chart_right = st.columns([1, 1])

        # Left Panel: Distribution of primary numerical column
        with chart_left:
            if num_cols:
                # Prefer Revenue or first numerical column
                preferred_col = next((c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower() or "amount" in c.lower()), num_cols[0])
                fig_dist = plot_numerical_distribution(active_df, preferred_col)
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No numerical features available for distribution analysis.")

        # Right Panel: Category breakdown of primary categorical column
        with chart_right:
            if cat_cols:
                preferred_cat = next((c for c in cat_cols if "cat" in c.lower() or "region" in c.lower() or "status" in c.lower()), cat_cols[0])
                fig_cat = plot_categorical_frequency(active_df, preferred_cat, top_n=6)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical features available for frequency analysis.")

    # =========================================================================
    # 5. BOTTOM SECTION: Capabilities & 3-Step Workflow
    # =========================================================================
    st.markdown(
        "<h3 style='font-size: 1.22rem; font-weight: 700; color: #F8FAFC; margin: 2rem 0 1rem 0; letter-spacing: -0.02em;'>Studio Capabilities</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #22D3EE;">📤</span> Data Ingestion &amp; Profiling
                </div>
                <p class="if-card-desc">
                    In-memory parsing of CSV and Excel spreadsheets with schema detection, type inference, and validation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #22D3EE;">🔍</span> Interactive Data Explorer
                </div>
                <p class="if-card-desc">
                    Multi-condition filtering, keyword search, dynamic sorting, column slicing, and retention metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #A855F7;">📈</span> Executive Business Analytics
                </div>
                <p class="if-card-desc">
                    Automated revenue KPIs, descriptive statistics, dispersion metrics, and Pearson correlation matrices.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #EC4899;">🚨</span> AI Anomaly Detection
                </div>
                <p class="if-card-desc">
                    Unsupervised Isolation Forest algorithm isolating multidimensional outliers and operational risk points.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #FBBF24;">💡</span> Contextual Smart Insights
                </div>
                <p class="if-card-desc">
                    Data-driven observations categorized by severity with statistical evidence and strategic recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="if-card">
                <div class="if-card-title">
                    <span style="color: #34D399;">📋</span> Multi-Format Export Center
                </div>
                <p class="if-card-desc">
                    Generate filtered UTF-8 CSV datasets and professional multi-sheet Excel workbooks with styled headers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Workflow Roadmap
    st.markdown(
        "<h3 style='font-size: 1.22rem; font-weight: 700; color: #F8FAFC; margin: 2rem 0 1rem 0; letter-spacing: -0.02em;'>Three-Step Intelligence Workflow</h3>",
        unsafe_allow_html=True,
    )

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            """
            <div class="if-step-card" style="border-top: 3px solid #22D3EE;">
                <span class="if-badge if-badge-cyan">Step 01</span>
                <div class="if-card-title" style="margin-top: 0.65rem;">Connect &amp; Profile</div>
                <p class="if-card-desc">
                    Upload your business dataset in CSV or Excel format. InsightFlow automatically validates schema completeness and detects column data types.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col2:
        st.markdown(
            """
            <div class="if-step-card" style="border-top: 3px solid #8B5CF6;">
                <span class="if-badge">Step 02</span>
                <div class="if-card-title" style="margin-top: 0.65rem;">Explore &amp; Experiment</div>
                <p class="if-card-desc">
                    Slice records with multi-dimensional filters, evaluate statistical distributions, benchmark groups, and run Isolation Forest anomaly scans.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col3:
        st.markdown(
            """
            <div class="if-step-card" style="border-top: 3px solid #EC4899;">
                <span class="if-badge if-badge-pink">Step 03</span>
                <div class="if-card-title" style="margin-top: 0.65rem;">Decide &amp; Export</div>
                <p class="if-card-desc">
                    Synthesize actionable recommendations, forecast time-series trajectories with uncertainty bands, and export publication-ready reports.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action Callout
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.info(
        "⚡ **Start Analysing:** Switch to **📤 Data Upload** in the sidebar to upload your business dataset, "
        "or select **sample_business_sales.csv** from the project directory to test all analytics routines."
    )
