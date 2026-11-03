"""Multi-sheet formatted Excel workbook generator for InsightFlow.

Generates professional .xlsx workbooks using Openpyxl with styling,
freeze panes, auto-width columns, and separate analytical sheets.
"""

import io
from typing import Any
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import compute_data_quality_summary, compute_column_quality_table
from src.analytics.kpi import compute_business_kpis
from src.analytics.insights import generate_smart_insights


# Styling constants
HEADER_FILL = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")  # Navy Blue
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
SECTION_FILL = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")  # Soft Blue
SECTION_FONT = Font(name="Calibri", size=11, bold=True, color="1E3A8A")
BODY_FONT = Font(name="Calibri", size=10, color="0F172A")
MUTED_FONT = Font(name="Calibri", size=10, italic=True, color="64748B")
THIN_BORDER = Border(
    left=Side(style="thin", color="E2E8F0"),
    right=Side(style="thin", color="E2E8F0"),
    top=Side(style="thin", color="E2E8F0"),
    bottom=Side(style="thin", color="E2E8F0"),
)


def _style_header_row(ws, row_idx: int = 1) -> None:
    """Apply professional header styling and freeze pane to a worksheet."""
    ws.row_dimensions[row_idx].height = 26
    for cell in ws[row_idx]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = f"A{row_idx + 1}"


def _autofit_columns(ws, min_width: int = 12, max_width: int = 60) -> None:
    """Adjust worksheet column dimensions according to maximum text length."""
    for col in ws.columns:
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len:
                max_len = len(val_str)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, min_width), max_width)


def _write_dataframe_to_sheet(ws, df: pd.DataFrame, start_row: int = 1) -> None:
    """Write DataFrame records and column headers into an openpyxl worksheet."""
    headers = list(df.columns)
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=start_row, column=col_idx, value=str(header))
        cell.border = THIN_BORDER

    for row_idx, row_values in enumerate(df.itertuples(index=False), start_row + 1):
        for col_idx, val in enumerate(row_values, 1):
            # Clean NaNs to blank strings for clean spreadsheet presentation
            clean_val = "" if pd.isna(val) else val
            cell = ws.cell(row=row_idx, column=col_idx, value=clean_val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER


def generate_excel_workbook_bytes(
    df: pd.DataFrame,
    include_raw_data: bool = True,
) -> bytes:
    """Generate a multi-sheet, presentation-grade Excel workbook.

    Sheets:
        1. Dataset (Active Raw Records)
        2. Statistical Summary (Descriptive metrics)
        3. Data Quality (Dataset and column integrity audit)
        4. Business KPIs (Inferred revenue, status distributions, primary metrics)
        5. Smart Insights (Automated observations and evidence)

    Returns:
        bytes representing the .xlsx workbook.
    """
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # 1. Dataset Sheet
    if include_raw_data and not df.empty:
        ws_data = wb.create_sheet(title="Dataset")
        # Ensure row count is within Excel's limits (1,048,576 rows)
        preview_df = df.iloc[:100000] if len(df) > 100000 else df
        _write_dataframe_to_sheet(ws_data, preview_df)
        _style_header_row(ws_data, 1)
        _autofit_columns(ws_data, min_width=12, max_width=45)

    # 2. Statistical Summary Sheet
    ws_stats = wb.create_sheet(title="Statistical Summary")
    stats_df = compute_numerical_statistics(df)
    if not stats_df.empty:
        # Reset index to write Feature column explicitly
        stats_to_write = stats_df.reset_index().rename(columns={"index": "Numerical Feature"})
        _write_dataframe_to_sheet(ws_stats, stats_to_write)
        _style_header_row(ws_stats, 1)
        _autofit_columns(ws_stats, min_width=14, max_width=25)
    else:
        ws_stats.cell(row=1, column=1, value="Notice").fill = SECTION_FILL
        ws_stats.cell(row=2, column=1, value="No numerical features identified in active dataset.")

    # 3. Data Quality Sheet
    ws_qual = wb.create_sheet(title="Data Quality")
    qual_summary = compute_data_quality_summary(df)

    # Write Quality KPI Block
    summary_headers = ["Metric", "Value"]
    summary_rows = [
        ("Total Rows", f"{qual_summary['total_rows']:,}"),
        ("Total Columns", f"{qual_summary['total_columns']:,}"),
        ("Overall Completeness", f"{qual_summary['completeness_pct']}%"),
        ("Missing Cells Count", f"{qual_summary['missing_cells']:,} ({qual_summary['missing_pct']}%)"),
        ("Duplicate Records Count", f"{qual_summary['duplicate_rows']:,} ({qual_summary['duplicate_pct']}%)"),
        ("Total Data Points", f"{qual_summary['total_cells']:,}"),
    ]

    for col_idx, h in enumerate(summary_headers, 1):
        c = ws_qual.cell(row=1, column=col_idx, value=h)
        c.fill = SECTION_FILL
        c.font = SECTION_FONT
        c.border = THIN_BORDER

    for r_idx, (m, v) in enumerate(summary_rows, 2):
        c_m = ws_qual.cell(row=r_idx, column=1, value=m)
        c_v = ws_qual.cell(row=r_idx, column=2, value=v)
        c_m.font = BODY_FONT
        c_v.font = BODY_FONT
        c_m.border = THIN_BORDER
        c_v.border = THIN_BORDER

    # Write Column-by-Column Table starting at row 9
    col_qual_df = compute_column_quality_table(df)
    if not col_qual_df.empty:
        _write_dataframe_to_sheet(ws_qual, col_qual_df, start_row=9)
        _style_header_row(ws_qual, 9)

    _autofit_columns(ws_qual, min_width=15, max_width=35)

    # 4. Business KPIs Sheet
    ws_kpi = wb.create_sheet(title="Business KPIs")
    kpis = compute_business_kpis(df)
    inferred = kpis.get("inferred_kpis", [])
    primary = kpis.get("primary_metric")
    status_df = kpis.get("status_distribution")

    kpi_records: list[dict[str, Any]] = []
    for k in inferred:
        kpi_records.append(
            {
                "KPI Dimension": k["name"],
                "Value": k["formatted"],
                "Methodology / Assumption": k["assumption"],
            }
        )

    if primary:
        kpi_records.append(
            {
                "KPI Dimension": f"Total {primary['column_name']}",
                "Value": f"{primary['sum']:,.2f}",
                "Methodology / Assumption": primary["assumption"],
            }
        )
        kpi_records.append(
            {
                "KPI Dimension": f"Average {primary['column_name']}",
                "Value": f"{primary['mean']:,.2f}",
                "Methodology / Assumption": primary["assumption"],
            }
        )

    if kpi_records:
        kpi_df = pd.DataFrame(kpi_records)
        _write_dataframe_to_sheet(ws_kpi, kpi_df)
        _style_header_row(ws_kpi, 1)

    # Append Status Distribution Table if present
    if status_df is not None and not status_df.empty:
        status_start = len(kpi_records) + 4
        ws_kpi.cell(row=status_start - 1, column=1, value="Status Distribution Breakdown").font = SECTION_FONT
        _write_dataframe_to_sheet(ws_kpi, status_df, start_row=status_start)
        _style_header_row(ws_kpi, status_start)

    _autofit_columns(ws_kpi, min_width=18, max_width=50)

    # 5. Smart Insights Sheet
    ws_ins = wb.create_sheet(title="Smart Insights")
    insights = generate_smart_insights(df)

    if insights:
        insight_rows = []
        for item in insights:
            # Evidence dictionary string
            ev_str = " | ".join([f"{k}: {v}" for k, v in item.get("evidence", {}).items()])
            insight_rows.append(
                {
                    "Severity": item.get("severity", "Info"),
                    "Category": item.get("category", "General"),
                    "Title": item.get("title", ""),
                    "Observation": item.get("description", ""),
                    "Supporting Evidence": ev_str,
                    "Why This Matters": item.get("why_it_matters", ""),
                }
            )

        ins_df = pd.DataFrame(insight_rows)
        _write_dataframe_to_sheet(ws_ins, ins_df)
        _style_header_row(ws_ins, 1)

        # Wrap text on Observation and Why It Matters columns
        for row in ws_ins.iter_rows(min_row=2):
            if len(row) >= 4:
                row[3].alignment = Alignment(wrap_text=True, vertical="top")
            if len(row) >= 6:
                row[5].alignment = Alignment(wrap_text=True, vertical="top")

        _autofit_columns(ws_ins, min_width=15, max_width=55)
    else:
        ws_ins.cell(row=1, column=1, value="Smart Insights").font = SECTION_FONT
        ws_ins.cell(row=2, column=1, value="No significant anomalies or imbalances detected.")

    # Save to in-memory bytes
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
