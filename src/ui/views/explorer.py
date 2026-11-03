"""Interactive Data Explorer view for InsightFlow.

Allows users to search, filter (categorical & numerical), sort, and project
columns interactively without mutating the underlying dataset.
"""

from typing import Any
import streamlit as st
import pandas as pd

from src.data.explorer import (
    classify_columns,
    search_dataset,
    filter_categorical,
    filter_numeric_range,
    sort_dataset,
    select_columns,
    get_numeric_bounds,
    get_unique_categories,
)


def render_explorer_view() -> None:
    """Render the Interactive Data Explorer interface."""
    # Ensure filter version state exists for clean reset functionality
    if "explorer_version" not in st.session_state:
        st.session_state["explorer_version"] = 0

    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # 1. Empty State if no dataset is active
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Interactive Data Explorer
                </h2>
                <p class="if-page-subtitle">
                    Search, filter, sort, and slice your tabular records interactively.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔍</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset to Explore
                </h3>
                <p style="margin: 0 auto; max-width: 480px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Please navigate to <strong>Data Upload</strong> in the sidebar to upload a CSV or Excel dataset first.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 2. Header and Reset Control
    filter_v = st.session_state["explorer_version"]
    filename = metadata["filename"] if metadata else "Active Dataset"

    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f"""
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Interactive Data Explorer
                </h2>
                <p class="if-page-subtitle">
                    Active Source: <strong style="color: #22D3EE;">{filename}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄  Reset All Filters", use_container_width=True):
            st.session_state["explorer_version"] += 1
            st.rerun()

    # Column classification
    classified = classify_columns(active_df)
    numerical_cols = classified["numerical"]
    categorical_cols = classified["categorical"]
    all_cols = [str(c) for c in active_df.columns]

    # 3. Global Search Input
    search_query = st.text_input(
        label="Global Dataset Search",
        placeholder="Type keywords to search across all columns (case-insensitive)...",
        key=f"exp_search_{filter_v}",
    )

    # 4. Multi-Dimensional Filter Controls
    with st.expander("🛠️  Filter, Sort & Column Projection Controls", expanded=True):
        f_col1, f_col2 = st.columns(2)

        # Categorical Filter
        with f_col1:
            st.markdown(
                "<p style='font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;'>Categorical Filter</p>",
                unsafe_allow_html=True,
            )
            cat_options = ["(None)"] + categorical_cols
            selected_cat_col = st.selectbox(
                label="Select Categorical Column",
                options=cat_options,
                key=f"cat_col_{filter_v}",
                label_visibility="collapsed",
            )

            selected_cat_values: list[Any] = []
            if selected_cat_col and selected_cat_col != "(None)":
                unique_vals = get_unique_categories(active_df, selected_cat_col)
                selected_cat_values = st.multiselect(
                    label=f"Filter values for '{selected_cat_col}'",
                    options=unique_vals,
                    default=[],
                    key=f"cat_vals_{filter_v}",
                    help="Leave blank to include all categories.",
                )

        # Numerical Range Filter
        with f_col2:
            st.markdown(
                "<p style='font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;'>Numerical Range Filter</p>",
                unsafe_allow_html=True,
            )
            num_options = ["(None)"] + numerical_cols
            selected_num_col = st.selectbox(
                label="Select Numerical Column",
                options=num_options,
                key=f"num_col_{filter_v}",
                label_visibility="collapsed",
            )

            slider_min, slider_max = 0.0, 1.0
            selected_range: tuple[float, float] | None = None
            if selected_num_col and selected_num_col != "(None)":
                slider_min, slider_max = get_numeric_bounds(active_df, selected_num_col)
                selected_range = st.slider(
                    label=f"Range for '{selected_num_col}'",
                    min_value=float(slider_min),
                    max_value=float(slider_max),
                    value=(float(slider_min), float(slider_max)),
                    key=f"num_range_{filter_v}",
                )

        st.markdown("<div class='if-divider' style='margin: 0.75rem 0;'></div>", unsafe_allow_html=True)

        # Sorting and Column Selection
        s_col1, s_col2, s_col3 = st.columns([2, 1, 3])

        with s_col1:
            sort_options = ["(None)"] + all_cols
            selected_sort_col = st.selectbox(
                label="Sort by Column",
                options=sort_options,
                key=f"sort_col_{filter_v}",
            )

        with s_col2:
            sort_direction = st.radio(
                label="Direction",
                options=["Ascending", "Descending"],
                horizontal=True,
                key=f"sort_dir_{filter_v}",
            )

        with s_col3:
            visible_cols = st.multiselect(
                label="Visible Columns",
                options=all_cols,
                default=all_cols,
                key=f"vis_cols_{filter_v}",
                help="Select which columns to display in the data table.",
            )

    # 5. Pipeline Execution (Preserves Original active_df)
    working_df = active_df.copy()

    # Step A: Apply global search
    if search_query:
        working_df = search_dataset(working_df, search_query)

    # Step B: Apply categorical filter
    if selected_cat_col and selected_cat_col != "(None)" and selected_cat_values:
        working_df = filter_categorical(working_df, selected_cat_col, selected_cat_values)

    # Step C: Apply numerical range filter
    if selected_num_col and selected_num_col != "(None)" and selected_range:
        working_df = filter_numeric_range(
            working_df,
            selected_num_col,
            min_val=selected_range[0],
            max_val=selected_range[1],
        )

    # Step D: Apply sorting
    if selected_sort_col and selected_sort_col != "(None)":
        is_asc = sort_direction == "Ascending"
        working_df = sort_dataset(working_df, selected_sort_col, ascending=is_asc)

    # Step E: Apply column projection
    display_cols = visible_cols if visible_cols else all_cols
    projected_df = select_columns(working_df, display_cols)
    st.session_state["filtered_dataset"] = projected_df

    # 6. Active Filter Indicators & Results Metrics Strip
    total_rows = len(active_df)
    filtered_rows = len(projected_df)
    retention_pct = (filtered_rows / total_rows * 100) if total_rows > 0 else 0.0

    active_filters: list[str] = []
    if search_query:
        active_filters.append(f"🔍 Search: *'{search_query}'*")
    if selected_cat_col and selected_cat_col != "(None)" and selected_cat_values:
        active_filters.append(f"🏷️ {selected_cat_col}: *{len(selected_cat_values)} selected*")
    if selected_num_col and selected_num_col != "(None)" and selected_range:
        active_filters.append(f"📊 {selected_num_col}: *[{selected_range[0]:,.1f} – {selected_range[1]:,.1f}]*")
    if selected_sort_col and selected_sort_col != "(None)":
        active_filters.append(f"🔃 Sort: *{selected_sort_col} ({sort_direction})*")
    if len(display_cols) < len(all_cols):
        active_filters.append(f"👁️ Columns: *{len(display_cols)} of {len(all_cols)}*")

    if active_filters:
        st.markdown(
            f"<div style='font-size: 0.84rem; color: #F8FAFC; margin: 0.6rem 0 0.4rem 0; padding: 0.55rem 0.95rem; background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(167, 139, 250, 0.28); border-radius: 10px;'>"
            f"<strong style='color: #22D3EE;'>Active Filters:</strong> &nbsp; {' &nbsp;&bull;&nbsp; '.join(active_filters)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Original Rows", value=f"{total_rows:,}")
    with m_col2:
        st.metric(label="Filtered Rows", value=f"{filtered_rows:,}")
    with m_col3:
        st.metric(label="Displayed Columns", value=f"{len(display_cols)} / {len(all_cols)}")
    with m_col4:
        st.metric(label="Match Rate", value=f"{retention_pct:.1f}%")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # 7. Interactive Filtered Table Display
    if projected_df.empty:
        st.warning(
            "⚠️ No records match the applied search and filter criteria. "
            "Try loosening filter conditions or click 'Reset All Filters' above."
        )
    else:
        st.dataframe(
            projected_df,
            use_container_width=True,
            height=460,
        )
