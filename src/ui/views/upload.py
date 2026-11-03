"""Data upload and ingestion view for InsightFlow.

Handles user file uploads (CSV, XLSX, XLS), validation, sheet selection,
dataset preview, and session state management.
"""

from typing import cast
import streamlit as st
import pandas as pd

from src.data.validator import (
    validate_file_extension,
    validate_file_size,
    validate_dataframe,
    extract_dataset_metadata,
)
from src.data.loader import (
    load_uploaded_dataset,
    inspect_excel_sheets,
)


def render_upload_view() -> None:
    """Render the Data Upload & Ingestion management interface."""
    # Ensure session state variables exist
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None
    if "dataset_metadata" not in st.session_state:
        st.session_state["dataset_metadata"] = None
    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    st.markdown(
        """
        <div class="if-page-header">
            <h2 class="if-page-title">
                Data Ingestion & Dataset Preview
            </h2>
            <p class="if-page-subtitle">
                Upload tabular business datasets in CSV or Excel format (up to 50 MB).
                Inspect column schemas, detect data types, and review initial records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Action Bar: Upload area + Clear Button
    col_upload, col_actions = st.columns([3, 1])

    with col_actions:
        st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
        clear_disabled = st.session_state["dataset"] is None
        if st.button("🗑️  Clear Dataset", disabled=clear_disabled, use_container_width=True):
            st.session_state["dataset"] = None
            st.session_state["dataset_metadata"] = None
            st.session_state["_active_file_signature"] = None
            st.session_state["uploader_version"] += 1
            st.rerun()

    with col_upload:
        uploader_key = f"uploader_{st.session_state['uploader_version']}"
        uploaded_file = st.file_uploader(
            label="Upload Business Dataset",
            type=["csv", "xlsx", "xls"],
            key=uploader_key,
            help="Supported formats: CSV, XLSX, XLS. Maximum file size: 50 MB.",
        )

    # First-Time User Ingestion Guidance
    with st.expander("💡 Ingestion Guidelines & Privacy Assurance", expanded=False):
        st.markdown(
            """
            - **Supported Formats:** CSV (`.csv`), Excel OpenXML (`.xlsx`), and legacy Excel (`.xls`).
            - **Size Limits:** Up to 50 MB per file.
            - **Encoding Fallback:** CSV files automatically decode UTF-8, Latin-1, or Windows-1252 character sets.
            - **Multi-Sheet Workbooks:** InsightFlow detects all sheets and allows you to select which sheet to analyze.
            - **In-Memory Privacy:** Files are parsed as in-memory byte streams; data is never transmitted externally or written to persistent disk storage.
            """
        )

    # Process uploaded file
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        file_size = len(file_bytes)

        # 1. Validate file extension
        ext_valid, ext_err = validate_file_extension(filename)
        if not ext_valid:
            st.error(f"❌ {ext_err}")
            return

        # 2. Validate file size (50 MB limit)
        size_valid, size_err = validate_file_size(file_size)
        if not size_valid:
            st.error(f"❌ {size_err}")
            return

        # 3. Handle multi-sheet Excel selection
        sheet_to_load: str | int | None = 0
        lower_name = filename.lower()
        if lower_name.endswith((".xlsx", ".xls")):
            available_sheets = inspect_excel_sheets(file_bytes)
            if len(available_sheets) > 1:
                st.info(f"Workbook contains {len(available_sheets)} sheets. Please select a sheet to analyze.")
                sheet_to_load = st.selectbox(
                    "Select Sheet",
                    options=available_sheets,
                    key=f"sheet_select_{uploader_key}",
                )
            elif len(available_sheets) == 1:
                sheet_to_load = available_sheets[0]

        # 4. Load dataset using in-memory parser
        with st.spinner("Ingesting and validating dataset..."):
            df, load_err = load_uploaded_dataset(
                file_bytes=file_bytes,
                filename=filename,
                sheet_name=sheet_to_load,
            )

        if load_err:
            st.error(f"❌ {load_err}")
            return

        # 5. Validate DataFrame content (empty checks, schema checks)
        df_valid, df_err = validate_dataframe(df)
        if not df_valid:
            st.error(f"❌ {df_err}")
            return

        # Successfully parsed: Store in session state and trigger immediate UI sync if new
        file_signature = f"{filename}_{file_size}_{sheet_to_load}"
        current_sig = st.session_state.get("_active_file_signature")

        if current_sig != file_signature:
            st.session_state["dataset"] = df
            st.session_state["dataset_metadata"] = extract_dataset_metadata(
                df=cast(pd.DataFrame, df),
                filename=filename,
                file_size_bytes=file_size,
                sheet_name=str(sheet_to_load) if sheet_to_load is not None else None,
            )
            st.session_state["_active_file_signature"] = file_signature
            st.rerun()

    # Render Active Dataset Details if available
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict | None = st.session_state.get("dataset_metadata")

    if active_df is not None and metadata is not None:
        st.markdown("<div class='if-divider'></div>", unsafe_allow_html=True)

        # Upload Confirmation Banner
        sheet_text = f" (Sheet: <strong>{metadata['sheet_name']}</strong>)" if metadata.get("sheet_name") else ""
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(52, 211, 153, 0.14) 0%, rgba(23, 23, 47, 0.85) 100%); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 14px; padding: 0.95rem 1.35rem; margin-bottom: 1.35rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
                <div>
                    <span style="color: #34D399; font-weight: 600; font-size: 0.94rem;">✓ Dataset Ingested Successfully</span>
                    <span style="color: #A5A5C5; font-size: 0.88rem; margin-left: 0.85rem;">
                        File: <strong style="color: #F8FAFC;">{metadata['filename']}</strong>{sheet_text} &bull; Size: <strong style="color: #F8FAFC;">{metadata['file_size_formatted']}</strong>
                    </span>
                </div>
                <span class="if-badge if-badge-success">Ready for Analytics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric summary cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(label="Total Rows", value=f"{metadata['total_rows']:,}")
        with metric_col2:
            st.metric(label="Total Columns", value=f"{metadata['total_columns']}")
        with metric_col3:
            st.metric(label="File Size", value=metadata["file_size_formatted"])
        with metric_col4:
            st.metric(label="Memory Footprint", value=metadata["memory_usage"])

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Data Preview Table (First 10 rows)
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <h3 style="margin: 0; font-size: 1.15rem; color: #F1F5F9; font-weight: 600;">
                    Dataset Preview (First 10 Rows)
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            active_df.head(10),
            use_container_width=True,
            height=380,
        )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # Column Schema & Data Types Breakdown
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <h3 style="margin: 0; font-size: 1.15rem; color: #F1F5F9; font-weight: 600;">
                    Detected Schema & Column Data Types
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        schema_df: pd.DataFrame = metadata["schema_table"]
        st.dataframe(
            schema_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        # Clean Empty State
        st.markdown(
            """
            <div class="if-placeholder-box" style="margin-top: 1rem;">
                <div style="font-size: 2.3rem; margin-bottom: 0.5rem;">📁</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.2rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset Loaded
                </h3>
                <p style="margin: 0 auto; max-width: 480px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Upload a CSV or Excel spreadsheet using the area above to begin schema profiling and exploratory analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
