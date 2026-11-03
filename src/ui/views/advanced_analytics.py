"""Advanced Analytics view for InsightFlow.

Renders high-level decision intelligence routines:
- Trend Analysis (period aggregation, growth rates, trajectory classification)
- Comparative Analysis (group benchmarks, secondary grouping, share %)
- Segmentation Analysis (categorical dimensions and numerical quantiles/ranges)
- Performance Rankings (top/bottom performers, tie handling, explanations)
- Distribution & Bivariate Relationship Analysis (box plots, scatter regression)
"""

from typing import Any
import streamlit as st
import pandas as pd
import numpy as np

from src.data.explorer import classify_columns
from src.analytics.advanced import (
    detect_date_columns,
    compute_trend_analysis,
    compute_comparative_analysis,
    compute_categorical_segmentation,
    compute_numerical_segmentation,
    compute_performance_rankings,
    compute_bivariate_relationship,
)
from src.visualization.advanced_charts import (
    plot_trend_analysis_chart,
    plot_comparative_chart,
    plot_segmentation_chart,
    plot_performance_ranking_chart,
    plot_grouped_box,
    plot_bivariate_scatter,
)


def render_advanced_analytics_view() -> None:
    """Render the Advanced Analytics interface."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # 1. Empty State Guard
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Advanced Analytics Engine
                </h2>
                <p class="if-page-subtitle">
                    High-level trend tracking, comparative benchmarks, multi-tier segmentation, and performance rankings.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔬</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset for Advanced Analytics
                </h3>
                <p style="margin: 0 auto; max-width: 500px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Please navigate to <strong>Data Upload</strong> in the sidebar to load a dataset first.
                    Once active, InsightFlow will enable time-series trajectory tracking, multi-dimensional
                    comparative benchmarks, quantile segmentation, and performance ranking diagnostics.
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
                Advanced Analytics Engine
            </h2>
            <p class="if-page-subtitle">
                Active Dataset: <strong style="color: #22D3EE;">{filename}</strong>
                &bull; {len(active_df):,} records &bull; {len(active_df.columns)} dimensions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Classify dimensions
    classified = classify_columns(active_df)
    numerical_cols = classified["numerical"]
    categorical_cols = classified["categorical"]
    date_cols = detect_date_columns(active_df)

    # 2. Main Feature Tabs
    tab_trend, tab_comp, tab_seg, tab_rank, tab_dist = st.tabs(
        [
            "📈  Trend Analysis",
            "⚖️  Comparative Analysis",
            "🧩  Segmentation Analysis",
            "🏆  Performance Analysis",
            "🔬  Distribution & Relationships",
        ]
    )

    # =========================================================================
    # TAB 1: Trend Analysis
    # =========================================================================
    with tab_trend:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Temporal Trend & Growth Analysis</h3>",
            unsafe_allow_html=True,
        )

        if not date_cols:
            st.info(
                "ℹ️ **No Date/Timestamp Column Detected:** Trend analysis requires at least one date or timestamp dimension. "
                "Ensure your dataset contains dates formatted like YYYY-MM-DD, ISO timestamps, or recognized year columns."
            )
        elif not numerical_cols:
            st.info("ℹ️ **No Numerical Columns Detected:** Trend analysis requires at least one numerical metric to evaluate changes over time.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sel_date = st.selectbox("Date Dimension", options=date_cols, key="trend_date_sel")
            with c2:
                sel_num = st.selectbox("Metric to Track", options=numerical_cols, key="trend_num_sel")
            with c3:
                freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D", "Yearly": "Y"}
                sel_freq_label = st.selectbox("Aggregation Period", options=list(freq_map.keys()), index=0, key="trend_freq_sel")
                sel_freq = freq_map[sel_freq_label]
            with c4:
                agg_map = {"Sum (Total)": "sum", "Mean (Average)": "mean", "Count (Records)": "count"}
                sel_agg_label = st.selectbox("Aggregation Metric", options=list(agg_map.keys()), index=0, key="trend_agg_sel")
                sel_agg = agg_map[sel_agg_label]

            trend_res = compute_trend_analysis(
                df=active_df,
                date_col=sel_date,
                metric_col=sel_num,
                frequency=sel_freq,
                agg_func=sel_agg,
            )

            if trend_res["status"] == "error":
                st.warning(f"⚠️ {trend_res['message']}")
            else:
                summary = trend_res["summary"]
                timeline_df = trend_res["timeline_df"]

                # Trajectory color & badge
                direction = summary.get("trend_direction", "Stable")
                dir_color = "#16A34A" if direction == "Increasing" else ("#DC2626" if direction == "Decreasing" else "#2563EB")
                growth_val = summary.get("overall_growth_pct")
                growth_str = f"{growth_val:+.1f}%" if growth_val is not None else f"{summary['total_change']:+,.2f}"

                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(label="Overall Trajectory", value=direction)
                with m2:
                    st.metric(label="Net Growth Rate", value=growth_str)
                with m3:
                    st.metric(label="Timeline Span", value=f"{summary['period_count']} {sel_freq_label}s")
                with m4:
                    st.metric(label="Period Average", value=f"{summary['average_value']:,.2f}")

                # Plotly Chart
                fig_trend = plot_trend_analysis_chart(
                    timeline_df=timeline_df,
                    date_col=sel_date,
                    metric_col=sel_num,
                    frequency_label=sel_freq_label,
                    average_val=summary["average_value"],
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # Narrative summary
                st.markdown(
                    f"""
                    <div class="if-callout">
                        <strong style="color: #22D3EE;">Trend Diagnostic:</strong> '{sel_num}' displays an <strong>{direction.lower()} trajectory</strong> across {summary['period_count']} {sel_freq_label.lower()} periods ({summary['start_period']} to {summary['end_period']}).
                        Baseline value began at <strong>{summary['start_value']:,.2f}</strong> and ended at <strong>{summary['end_value']:,.2f}</strong> (net shift: <strong>{growth_str}</strong>).
                        The historical peak occurred in <strong>{summary['peak_period']}</strong> ({summary['peak_value']:,.2f}), and the trough occurred in <strong>{summary['trough_period']}</strong> ({summary['trough_value']:,.2f}).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Detailed timeline table
                with st.expander(f"📋 View Full {sel_freq_label} Timeline Breakdown ({len(timeline_df)} periods)", expanded=False):
                    disp_timeline = timeline_df[["period", "value", "change", "pct_change"]].copy()
                    disp_timeline.columns = ["Period", f"{sel_agg_label} {sel_num}", "Net Change", "Period Growth (%)"]
                    st.dataframe(disp_timeline, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Comparative Analysis
    # =========================================================================
    with tab_comp:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Cross-Category Comparative Benchmarking</h3>",
            unsafe_allow_html=True,
        )

        if not categorical_cols:
            st.info("ℹ️ **No Categorical Columns Detected:** Comparative analysis requires at least one grouping category (e.g. Region, Category, Department).")
        elif not numerical_cols:
            st.info("ℹ️ **No Numerical Columns Detected:** Comparative analysis requires at least one numerical metric to aggregate across groups.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                sel_grp = st.selectbox("Primary Grouping Dimension", options=categorical_cols, key="comp_grp_sel")
            with cc2:
                sel_comp_num = st.selectbox("Target Comparison Metric", options=numerical_cols, key="comp_num_sel")
            with cc3:
                sec_opts = ["(None - Single Level)"] + [c for c in categorical_cols if c != sel_grp]
                sel_sec = st.selectbox("Secondary Grouping (Optional Cross-Tab)", options=sec_opts, key="comp_sec_sel")
                actual_sec = None if sel_sec.startswith("(None") else sel_sec

            comp_res = compute_comparative_analysis(
                df=active_df,
                group_col=sel_grp,
                metric_col=sel_comp_num,
                secondary_col=actual_sec,
            )

            if comp_res["status"] == "error":
                st.warning(f"⚠️ {comp_res['message']}")
            else:
                meta = comp_res["metadata"]
                summary_table = comp_res["summary_table"]
                pivot_df = comp_res["pivot_table"]

                # Metric Cards
                cm1, cm2, cm3, cm4 = st.columns(4)
                with cm1:
                    st.metric(label="Distinct Groups", value=f"{meta['group_count']:,}")
                with cm2:
                    st.metric(label=f"Top Group: {meta['top_group']['name']}", value=f"{meta['top_group']['total']:,.2f}")
                with cm3:
                    st.metric(label="Top Group Share", value=f"{meta['top_group']['share_pct']:.1f}%")
                with cm4:
                    st.metric(label="Total Aggregated", value=f"{meta['overall_total']:,.2f}")

                # Plotly Chart
                fig_comp = plot_comparative_chart(
                    comp_df=summary_table,
                    group_col=sel_grp,
                    metric_col=sel_comp_num,
                    secondary_col=actual_sec,
                    pivot_df=pivot_df if actual_sec else None,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Summary Table
                st.markdown(
                    f"<h4 style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin: 1rem 0 0.5rem 0;'>Statistical Breakdown by {sel_grp}</h4>",
                    unsafe_allow_html=True,
                )
                st.dataframe(summary_table, use_container_width=True, hide_index=True)

                if actual_sec and not pivot_df.empty:
                    with st.expander(f"📊 Cross-Tabulation Matrix ({sel_grp} × {actual_sec})", expanded=False):
                        st.dataframe(pivot_df, use_container_width=True)

    # =========================================================================
    # TAB 3: Segmentation Analysis
    # =========================================================================
    with tab_seg:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Dataset Segmentation & Tier Profiling</h3>",
            unsafe_allow_html=True,
        )

        seg_mode = st.radio(
            "Select Segmentation Strategy",
            options=["Categorical Dimension Segmentation", "Numerical Quantile / Range Tiers"],
            horizontal=True,
            key="seg_strategy_sel",
        )

        if seg_mode == "Categorical Dimension Segmentation":
            if not categorical_cols:
                st.info("ℹ️ No categorical columns available for categorical segmentation.")
            else:
                sg_c1, sg_c2 = st.columns(2)
                with sg_c1:
                    sel_seg_dim = st.selectbox("Segmentation Category", options=categorical_cols, key="seg_dim_sel")
                with sg_c2:
                    num_opts = ["(Record Counts Only)"] + numerical_cols
                    sel_seg_metric_label = st.selectbox("Supporting Metric (Optional)", options=num_opts, key="seg_metric_sel")
                    actual_seg_metric = None if sel_seg_metric_label.startswith("(Record") else sel_seg_metric_label

                seg_res = compute_categorical_segmentation(
                    df=active_df,
                    segment_col=sel_seg_dim,
                    metric_col=actual_seg_metric,
                )

                if seg_res["status"] == "error":
                    st.warning(f"⚠️ {seg_res['message']}")
                else:
                    seg_meta = seg_res["metadata"]
                    seg_table = seg_res["segment_table"]

                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.metric(label="Total Segments", value=f"{seg_meta['segment_count']}")
                    with sm2:
                        st.metric(label=f"Dominant: {seg_meta['dominant_segment']['name']}", value=f"{seg_meta['dominant_segment']['records']:,} rows")
                    with sm3:
                        st.metric(label="Dominant Share", value=f"{seg_meta['dominant_segment']['share_pct']:.1f}%")
                    with sm4:
                        st.metric(label=f"Smallest: {seg_meta['smallest_segment']['name']}", value=f"{seg_meta['smallest_segment']['records']:,} rows")

                    # Donut Chart
                    fig_seg = plot_segmentation_chart(
                        seg_df=seg_table,
                        label_col=sel_seg_dim,
                        count_col="Records",
                        title=f"Segment Proportions by {sel_seg_dim}",
                    )
                    st.plotly_chart(fig_seg, use_container_width=True)

                    st.dataframe(seg_table, use_container_width=True, hide_index=True)

        else:  # Numerical Quantile / Range Tiers
            if not numerical_cols:
                st.info("ℹ️ No numerical columns available to create numerical tier segments.")
            else:
                nq1, nq2, nq3, nq4 = st.columns(4)
                with nq1:
                    sel_tier_col = st.selectbox("Numerical Feature to Segment", options=numerical_cols, key="tier_col_sel")
                with nq2:
                    tier_method = st.selectbox("Segmentation Method", options=["quantiles (Equal Frequency)", "ranges (Equal Width)"], key="tier_method_sel")
                    actual_method = "quantiles" if "quantiles" in tier_method else "ranges"
                with nq3:
                    tier_bins = st.slider("Number of Tiers", min_value=2, max_value=8, value=4, key="tier_bins_sel")
                with nq4:
                    t_opts = ["(None)"] + [c for c in numerical_cols if c != sel_tier_col]
                    sel_t_metric = st.selectbox("Cross-Metric Summary", options=t_opts, key="tier_t_metric_sel")
                    actual_t_metric = None if sel_t_metric == "(None)" else sel_t_metric

                tier_res = compute_numerical_segmentation(
                    df=active_df,
                    num_col=sel_tier_col,
                    method=actual_method,
                    num_bins=tier_bins,
                    target_metric=actual_t_metric,
                )

                if tier_res["status"] == "error":
                    st.warning(f"⚠️ {tier_res['message']}")
                else:
                    tier_meta = tier_res["metadata"]
                    tier_table = tier_res["segment_table"]

                    tm1, tm2, tm3, tm4 = st.columns(4)
                    with tm1:
                        st.metric(label="Tiers Created", value=f"{tier_meta['bins_created']}")
                    with tm2:
                        st.metric(label=f"Dominant: {tier_meta['dominant_segment']['name']}", value=f"{tier_meta['dominant_segment']['records']:,} rows")
                    with tm3:
                        st.metric(label="Dominant Tier Share", value=f"{tier_meta['dominant_segment']['share_pct']:.1f}%")
                    with tm4:
                        st.metric(label=f"Smallest: {tier_meta['smallest_segment']['name']}", value=f"{tier_meta['smallest_segment']['records']:,} rows")

                    fig_tier = plot_segmentation_chart(
                        seg_df=tier_table,
                        label_col="segment",
                        count_col="Records",
                        title=f"Numerical Tier Proportions: {sel_tier_col} ({tier_meta['method'].capitalize()})",
                    )
                    st.plotly_chart(fig_tier, use_container_width=True)

                    st.dataframe(tier_table, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: Performance Analysis
    # =========================================================================
    with tab_rank:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Leaderboard & Performance Ranking Diagnostics</h3>",
            unsafe_allow_html=True,
        )

        if not categorical_cols:
            st.info("ℹ️ Performance ranking requires a categorical grouping column.")
        elif not numerical_cols:
            st.info("ℹ️ Performance ranking requires a numerical score metric.")
        else:
            rk1, rk2, rk3, rk4 = st.columns(4)
            with rk1:
                sel_rk_grp = st.selectbox("Entity / Category to Rank", options=categorical_cols, key="rk_grp_sel")
            with rk2:
                sel_rk_num = st.selectbox("Ranking Metric", options=numerical_cols, key="rk_num_sel")
            with rk3:
                rk_agg_map = {"Total Volume (Sum)": "sum", "Average Intensity (Mean)": "mean"}
                sel_rk_agg_label = st.selectbox("Scoring Method", options=list(rk_agg_map.keys()), key="rk_agg_sel")
                actual_rk_agg = rk_agg_map[sel_rk_agg_label]
            with rk4:
                sel_top_n = st.slider("Top / Bottom N Cutoff", min_value=3, max_value=15, value=5, key="rk_top_n_sel")

            rank_res = compute_performance_rankings(
                df=active_df,
                group_col=sel_rk_grp,
                metric_col=sel_rk_num,
                agg_func=actual_rk_agg,
                top_n=sel_top_n,
            )

            if rank_res["status"] == "error":
                st.warning(f"⚠️ {rank_res['message']}")
            else:
                rk_meta = rank_res["metadata"]
                ranked_table = rank_res["ranked_table"]
                top_p = rank_res["top_performers"]
                bottom_p = rank_res["bottom_performers"]

                rm1, rm2, rm3, rm4 = st.columns(4)
                with rm1:
                    st.metric(label=f"🥇 #1 Rank: {rk_meta['top_performer']['name']}", value=f"{rk_meta['top_performer']['score']:,.2f}")
                with rm2:
                    st.metric(label="#1 Share of Total", value=f"{rk_meta['top_performer']['share_pct']:.1f}%")
                with rm3:
                    st.metric(label=f"⚠️ Lowest: {rk_meta['bottom_performer']['name']}", value=f"{rk_meta['bottom_performer']['score']:,.2f}")
                with rm4:
                    st.metric(label="Total Ranked Entities", value=f"{rk_meta['total_groups']:,}")

                # Ranking Chart
                fig_rank = plot_performance_ranking_chart(
                    top_df=top_p,
                    bottom_df=bottom_p,
                    group_col=sel_rk_grp,
                    metric_label=rk_meta["metric_label"],
                )
                st.plotly_chart(fig_rank, use_container_width=True)

                # Narrative explanation
                st.info(f"💡 **Evaluation Methodology:** {rk_meta['explanation']}")

                # Side by Side Top N and Bottom N
                col_top, col_bot = st.columns(2)
                with col_top:
                    st.markdown(f"#### 🏆 Top {len(top_p)} Performers")
                    st.dataframe(top_p, use_container_width=True, hide_index=True)
                with col_bot:
                    st.markdown(f"#### ⚠️ Bottom {len(bottom_p)} Performers")
                    st.dataframe(bottom_p, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5: Distribution & Relationships
    # =========================================================================
    with tab_dist:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Distribution Spread & Bivariate Regression</h3>",
            unsafe_allow_html=True,
        )

        sub_mode = st.radio(
            "Select Diagnostic Mode",
            options=["Grouped Distribution Comparison (Box Plot)", "Bivariate Feature Relationship (Scatter & Linear Fit)"],
            horizontal=True,
            key="dist_sub_mode_sel",
        )

        if sub_mode == "Grouped Distribution Comparison (Box Plot)":
            if not categorical_cols or not numerical_cols:
                st.info("ℹ️ Grouped distribution comparison requires at least one categorical dimension and one numerical metric.")
            else:
                bx1, bx2 = st.columns(2)
                with bx1:
                    sel_bx_cat = st.selectbox("Category Dimension", options=categorical_cols, key="bx_cat_sel")
                with bx2:
                    sel_bx_num = st.selectbox("Numerical Metric", options=numerical_cols, key="bx_num_sel")

                fig_box = plot_grouped_box(active_df, cat_col=sel_bx_cat, num_col=sel_bx_num)
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption(f"Displays quartiles, median markers, and statistical outliers across the top categories in '{sel_bx_cat}'.")

        else:  # Bivariate Feature Relationship
            if len(numerical_cols) < 2:
                st.info("ℹ️ Bivariate relationship analysis requires at least 2 numerical features to evaluate correlation and regression.")
            else:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    sel_sc_x = st.selectbox("Independent Feature (X-Axis)", options=numerical_cols, index=0, key="sc_x_sel")
                with sc2:
                    y_opts = [c for c in numerical_cols if c != sel_sc_x]
                    sel_sc_y = st.selectbox("Dependent Feature (Y-Axis)", options=y_opts if y_opts else numerical_cols, key="sc_y_sel")
                with sc3:
                    cat_opts = ["(None)"] + categorical_cols
                    sel_sc_color = st.selectbox("Color Grouping (Optional)", options=cat_opts, key="sc_color_sel")
                    actual_sc_color = None if sel_sc_color == "(None)" else sel_sc_color

                rel_res = compute_bivariate_relationship(
                    df=active_df,
                    x_col=sel_sc_x,
                    y_col=sel_sc_y,
                )

                if rel_res["status"] == "error":
                    st.warning(f"⚠️ {rel_res['message']}")
                else:
                    rel_metrics = rel_res["metrics"]

                    bm1, bm2, bm3, bm4 = st.columns(4)
                    with bm1:
                        st.metric(label="Pearson Correlation (r)", value=f"{rel_metrics['pearson_r']:+.3f}")
                    with bm2:
                        st.metric(label="Goodness of Fit (R²)", value=f"{rel_metrics['r_squared']:.3f}")
                    with bm3:
                        st.metric(label="Relationship Strength", value=rel_metrics["relationship"])
                    with bm4:
                        st.metric(label="Analyzed Observations", value=f"{rel_metrics['sample_size']:,}")

                    fig_scatter = plot_bivariate_scatter(
                        df=active_df,
                        x_col=sel_sc_x,
                        y_col=sel_sc_y,
                        color_col=actual_sc_color,
                        trend_metrics=rel_metrics,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="if-callout">
                            <strong style="color: #22D3EE;">Regression Model:</strong> <code>{rel_metrics['equation']}</code><br>
                            The model explains <strong>{rel_metrics['r_squared']*100:.1f}%</strong> of variance in '{sel_sc_y}'.
                            For every unit increase in '{sel_sc_x}', '{sel_sc_y}' changes on average by <strong>{rel_metrics['slope']:+,.4f}</strong> units.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
