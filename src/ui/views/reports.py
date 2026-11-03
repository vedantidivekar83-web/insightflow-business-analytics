"""Executive Reports and Export dashboard view for InsightFlow.

Allows users to export active and filtered datasets as UTF-8 CSVs,
generate presentation-grade multi-sheet Excel workbooks (.xlsx),
and reviews PDF executive report availability.
"""

from typing import Any
import datetime
import streamlit as st
import pandas as pd

from src.reports.csv_export import generate_csv_bytes
from src.reports.excel_export import generate_excel_workbook_bytes
from src.reports.pdf_export import is_pdf_export_available, get_pdf_limitation_message


def render_reports_view() -> None:
    """Render the Reports & Data Export interface."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")
    filtered_df: pd.DataFrame | None = st.session_state.get("filtered_dataset")

    # 1. Empty State Guard
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Reports & Export Center
                </h2>
                <p class="if-page-subtitle">
                    Export raw and filtered datasets, generate presentation-grade workbooks, and compile analytical briefs.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📋</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset to Export
                </h3>
                <p style="margin: 0 auto; max-width: 500px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Please navigate to <strong>Data Upload</strong> in the sidebar to load a dataset before generating exports.
                    Once active, you will be able to download cleaned CSV files, structured multi-sheet Excel workbooks,
                    and analytical summaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filename = metadata["filename"] if metadata else "dataset.csv"
    base_name = filename.rsplit(".", 1)[0]
    today_str = datetime.date.today().strftime("%Y%m%d")

    # Header section
    st.markdown(
        f"""
        <div class="if-page-header">
            <h2 class="if-page-title">
                Reports & Export Center
            </h2>
            <p class="if-page-subtitle">
                Active Dataset: <strong style="color: #22D3EE;">{filename}</strong>
                &bull; {len(active_df):,} records ready for export
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Summary Strip
    total_rows = len(active_df)
    total_cols = len(active_df.columns)
    has_filtered = filtered_df is not None and len(filtered_df) != total_rows
    filtered_rows = len(filtered_df) if filtered_df is not None else total_rows

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric(label="Total Records", value=f"{total_rows:,}")
    with s_col2:
        st.metric(label="Total Dimensions", value=f"{total_cols}")
    with s_col3:
        st.metric(label="Explorer Filtered Rows", value=f"{filtered_rows:,}")
    with s_col4:
        st.metric(label="Active Export Status", value="Ready")

    st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.85rem;'>Available Export Formats</h3>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Format 1: CSV Export
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 📄 Comma-Separated Values (CSV)")
        st.write(
            "Export tabular data in universal UTF-8 encoded CSV format. "
            "Compatible with data warehouses, spreadsheets, and SQL databases."
        )

        csv_col1, csv_col2 = st.columns(2)

        # Download Option 1: Full active dataset
        with csv_col1:
            full_csv_bytes = generate_csv_bytes(active_df)
            full_csv_filename = f"InsightFlow_{base_name}_full_{today_str}.csv"
            st.download_button(
                label="📥  Download Full Dataset (CSV)",
                data=full_csv_bytes,
                file_name=full_csv_filename,
                mime="text/csv",
                use_container_width=True,
                help=f"Exports all {total_rows:,} records and {total_cols} columns.",
            )
            st.caption(f"Full dataset ({total_rows:,} rows • {total_cols} cols)")

        # Download Option 2: Filtered Data Explorer dataset
        with csv_col2:
            if filtered_df is not None:
                filtered_csv_bytes = generate_csv_bytes(filtered_df)
                filtered_csv_filename = f"InsightFlow_{base_name}_filtered_{today_str}.csv"
                st.download_button(
                    label="📥  Download Filtered Explorer Dataset (CSV)",
                    data=filtered_csv_bytes,
                    file_name=filtered_csv_filename,
                    mime="text/csv",
                    use_container_width=True,
                    help=f"Exports the {len(filtered_df):,} rows currently sliced in Data Explorer.",
                )
                st.caption(f"Filtered subset ({len(filtered_df):,} rows • {len(filtered_df.columns)} cols)")
            else:
                st.button("📥  Download Filtered Dataset (CSV)", disabled=True, use_container_width=True)
                st.caption("Apply filters in Data Explorer first to enable subset export.")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Format 2: Multi-Sheet Excel Workbook (.xlsx)
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 📊 Multi-Sheet Excel Workbook (.xlsx)")
        st.write(
            "Generates a multi-sheet, formatted Excel workbook using Openpyxl. "
            "Includes dedicated sheets for raw records, descriptive statistics, data quality audits, "
            "business KPIs, and synthesized Smart Insights."
        )

        st.markdown(
            """
            - **Sheet 1 (`Dataset`):** Complete active tabular records with frozen headers.
            - **Sheet 2 (`Statistical Summary`):** Mean, median, mode, dispersion, and quartiles.
            - **Sheet 3 (`Data Quality`):** Completeness score, missing values audit, and duplicate counts.
            - **Sheet 4 (`Business KPIs`):** Inferred revenue, AOV, order status distributions, and primary metrics.
            - **Sheet 5 (`Smart Insights`):** All automated observations, evidence values, and contextual explanations.
            """
        )

        excel_filename = f"InsightFlow_{base_name}_Report_{today_str}.xlsx"

        with st.spinner("Compiling multi-sheet Excel workbook..."):
            excel_bytes = generate_excel_workbook_bytes(active_df)

        st.download_button(
            label="📥  Download Multi-Sheet Excel Report (.xlsx)",
            data=excel_bytes,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Format 3: PDF Executive Report
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 📑 Executive PDF Report")
        st.write(
            "Compiles a formal executive summary document containing branding, high-level metrics, "
            "data quality findings, and strategic smart insights."
        )

        if is_pdf_export_available():
            st.success("PDF generation engine is active.")
            # PDF generation placeholder
        else:
            st.info(f"ℹ️ **Format Status:** {get_pdf_limitation_message()}")
            st.button(
                label="🔒  PDF Export (Requires 'reportlab' Approval)",
                disabled=True,
                use_container_width=True,
            )
