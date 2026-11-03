"""Comprehensive verification script testing all 8 InsightFlow views, analytics, and exports."""

import sys
import os
import io
import datetime
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# 1. Create a rich sample dataset
np.random.seed(42)
n_rows = 150
dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")
categories = ["Electronics", "Software", "Cloud Services", "Consulting", "Hardware"]
regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
statuses = ["Completed", "Shipped", "Pending", "Cancelled"]

sample_df = pd.DataFrame(
    {
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n_rows)],
        "Date": dates.strftime("%Y-%m-%d"),
        "Category": np.random.choice(categories, size=n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "Region": np.random.choice(regions, size=n_rows),
        "Quantity": np.random.randint(1, 25, size=n_rows),
        "Unit_Price": np.round(np.random.uniform(50.0, 950.0, size=n_rows), 2),
        "Discount": np.round(np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2], size=n_rows), 2),
        "Customer_Score": np.random.randint(1, 6, size=n_rows),
        "Order_Status": np.random.choice(statuses, size=n_rows, p=[0.65, 0.2, 0.1, 0.05]),
    }
)
sample_df["Revenue"] = np.round(sample_df["Quantity"] * sample_df["Unit_Price"] * (1 - sample_df["Discount"]), 2)

# Save sample dataset to disk for manual testing
sample_csv_path = os.path.abspath("sample_business_sales.csv")
sample_df.to_csv(sample_csv_path, index=False)
print(f"[OK] Sample dataset saved to: {sample_csv_path} ({len(sample_df)} rows, {len(sample_df.columns)} columns)")

from src.data.validator import extract_dataset_metadata, validate_dataframe
from src.data.explorer import (
    classify_columns,
    search_dataset,
    filter_categorical,
    filter_numeric_range,
    sort_dataset,
    select_columns,
)
from src.analytics.kpi import compute_business_kpis
from src.analytics.statistics import compute_numerical_statistics
from src.analytics.quality import compute_data_quality_summary, compute_column_quality_table
from src.analytics.insights import generate_smart_insights
from src.analytics.advanced import (
    compute_trend_analysis,
    compute_comparative_analysis,
    compute_categorical_segmentation,
    compute_numerical_segmentation,
    compute_performance_rankings,
    compute_bivariate_relationship,
)
from src.ml.anomaly_detection import detect_anomalies
from src.ml.prediction import infer_ml_task, train_predictive_model
from src.ml.forecasting import generate_time_series_forecast
from src.reports.csv_export import generate_csv_bytes
from src.reports.excel_export import generate_excel_workbook_bytes
from src.visualization.charts import (
    plot_numerical_distribution,
    plot_categorical_frequency,
    plot_correlation_heatmap,
    plot_time_series_trend,
)

# 2. Test Ingestion and Metadata
v_ok, v_err = validate_dataframe(sample_df)
assert v_ok, f"Validation failed: {v_err}"
metadata = extract_dataset_metadata(sample_df, "sample_business_sales.csv", os.path.getsize(sample_csv_path))
print(f"[OK] Metadata extracted: {metadata['total_rows']} rows, {metadata['total_columns']} dimensions, size: {metadata['file_size_formatted']}")

# 3. Test Explorer
res_search = search_dataset(sample_df, "Electronics")
assert len(res_search) > 0, "Search query returned 0 rows"
res_cat = filter_categorical(sample_df, "Category", ["Electronics", "Software"])
assert len(res_cat) > 0, "Categorical filter returned 0 rows"
res_num = filter_numeric_range(sample_df, "Quantity", 5, 20)
assert len(res_num) > 0, "Numeric range filter returned 0 rows"
res_sort = sort_dataset(sample_df, "Revenue", ascending=False)
assert res_sort["Revenue"].iloc[0] >= res_sort["Revenue"].iloc[-1], "Sort failed"
print("[OK] Data Explorer: Search, Categorical, Numeric, and Sorting verified")

# 4. Test Analytics
kpi_res = compute_business_kpis(sample_df)
assert kpi_res["total_records"] == 150
stats_res = compute_numerical_statistics(sample_df)
assert not stats_res.empty, "Stats dataframe is empty"
qual_res = compute_data_quality_summary(sample_df)
assert qual_res["completeness_pct"] == 100.0, "Completeness mismatch"
print("[OK] Analytics: KPIs, Descriptive Statistics, and Data Quality verified")

# 5. Test Smart Insights
insights_res = generate_smart_insights(sample_df)
assert len(insights_res) > 0, "No insights generated"
print(f"[OK] Smart Insights: Generated {len(insights_res)} data-driven observations")

# 6. Test Advanced Analytics
trend_res = compute_trend_analysis(sample_df, "Date", "Revenue", "M")
assert trend_res["status"] == "ok", f"Trend analysis failed: {trend_res.get('message')}"
comp_res = compute_comparative_analysis(sample_df, "Region", "Revenue", "sum")
assert comp_res["status"] == "ok", f"Comparative analysis failed: {comp_res.get('message')}"
seg_res = compute_categorical_segmentation(sample_df, "Category", "Revenue")
assert seg_res["status"] == "ok", f"Segmentation failed: {seg_res.get('message')}"
rank_res = compute_performance_rankings(sample_df, "Category", "Revenue", top_n=3)
assert rank_res["status"] == "ok", f"Ranking failed: {rank_res.get('message')}"
biv_res = compute_bivariate_relationship(sample_df, "Unit_Price", "Revenue")
assert biv_res["status"] == "ok", f"Bivariate relationship failed: {biv_res.get('message')}"
print("[OK] Advanced Analytics: Trend, Comparative, Segmentation, Ranking & Bivariate verified")

# 7. Test Machine Learning
anom_res = detect_anomalies(sample_df, ["Quantity", "Unit_Price", "Revenue"], contamination=0.05)
assert anom_res["status"] == "ok", f"Anomaly detection failed: {anom_res.get('message')}"
train_res = train_predictive_model(sample_df, "Order_Status", ["Quantity", "Unit_Price", "Revenue"])
assert train_res["status"] == "ok", f"Predictive model failed: {train_res.get('message')}"
fc_res = generate_time_series_forecast(sample_df, "Date", "Revenue", forecast_periods=7)
assert fc_res["status"] == "ok", f"Forecast failed: {fc_res.get('message')}"
print("[OK] Machine Learning: Anomaly Detection, Supervised Classifier, Time-Series Forecast verified")

# 8. Test Reports & Exports
csv_bytes = generate_csv_bytes(sample_df)
assert len(csv_bytes) > 0, "CSV bytes empty"
xlsx_bytes = generate_excel_workbook_bytes(sample_df)
assert len(xlsx_bytes) > 0, "Excel bytes empty"
print(f"[OK] Reports: CSV ({len(csv_bytes):,} bytes), Excel Workbook ({len(xlsx_bytes):,} bytes) verified")

# 9. Test View Rendering across all 8 pages
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

mock_state = {
    "dataset": sample_df,
    "dataset_metadata": metadata,
    "filtered_dataset": sample_df,
    "uploader_version": 0,
    "explorer_version": 0,
}

print("\n--- Verifying View Execution with Mocked State ---")
with patch("streamlit.session_state", mock_state), \
     patch("streamlit.sidebar") as mock_sb, \
     patch("streamlit.radio", return_value="🏠  Overview"):
    mock_sb.__enter__ = MagicMock(return_value=mock_sb)
    mock_sb.__exit__ = MagicMock(return_value=None)

    render_header()
    print("[OK] Page Header rendered cleanly")

    render_sidebar()
    print("[OK] Sidebar navigation rendered cleanly")

    render_overview_view()
    print("[OK] 1. Overview Page rendered cleanly")

    render_upload_view()
    print("[OK] 2. Data Upload Page rendered cleanly")

    render_explorer_view()
    print("[OK] 3. Data Explorer Page rendered cleanly")

    render_analytics_view()
    print("[OK] 4. Business Analytics Page rendered cleanly")

    render_advanced_analytics_view()
    print("[OK] 5. Advanced Analytics Page rendered cleanly")

    render_ml_insights_view()
    print("[OK] 6. Machine Learning Page rendered cleanly")

    render_insights_view()
    print("[OK] 7. Smart Insights Page rendered cleanly")

    render_reports_view()
    print("[OK] 8. Reports & Export Page rendered cleanly")

print("\n=== ALL 8 VIEWS, WORKFLOWS, ALGORITHMS & EXPORTS VERIFIED SUCCESSFULLY ===")
