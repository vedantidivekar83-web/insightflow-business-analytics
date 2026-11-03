"""Smart Insights engine for InsightFlow.

Synthesizes data quality audits, statistical summaries, correlation matrices,
time trends, and domain-inferred KPIs into actionable, data-backed observations.
"""

from typing import Any
import pandas as pd
import numpy as np

from src.analytics.quality import (
    compute_data_quality_summary,
    compute_column_quality_table,
)
from src.analytics.statistics import compute_numerical_statistics
from src.analytics.kpi import compute_business_kpis
from src.analytics.correlation import (
    compute_correlation_matrix,
    get_top_correlations,
)
from src.data.explorer import classify_columns


def generate_quality_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate observations on data completeness, missingness, and duplication."""
    if df.empty:
        return []

    insights: list[dict[str, Any]] = []
    summary = compute_data_quality_summary(df)
    col_table = compute_column_quality_table(df)

    # 1. Duplicate records
    dup_count = summary["duplicate_rows"]
    dup_pct = summary["duplicate_pct"]
    if dup_count > 0:
        severity = "Warning" if dup_pct < 10.0 else "Critical"
        insights.append(
            {
                "id": "qual_duplicate_rows",
                "category": "Data Quality",
                "severity": severity,
                "title": f"Duplicate Records Detected ({dup_count:,} rows)",
                "description": (
                    f"The dataset contains {dup_count:,} duplicate rows, representing {dup_pct:.1f}% "
                    f"of all records."
                ),
                "evidence": {
                    "Duplicate Rows": f"{dup_count:,}",
                    "Duplication Share": f"{dup_pct:.1f}%",
                    "Total Rows": f"{summary['total_rows']:,}",
                },
                "why_it_matters": (
                    "Duplicate entries can artificially inflate transactional volume, skew statistical averages, "
                    "and distort customer or product aggregations."
                ),
            }
        )

    # 2. Significant column missingness (> 10% missing)
    if not col_table.empty:
        high_missing = col_table[col_table["Missing (%)"] >= 10.0]
        for _, row in high_missing.iterrows():
            col_name = str(row["Column"])
            miss_count = int(row["Missing Count"])
            miss_pct = float(row["Missing (%)"])
            severity = "Critical" if miss_pct >= 30.0 else "Warning"

            insights.append(
                {
                    "id": f"qual_missing_{col_name}",
                    "category": "Data Quality",
                    "severity": severity,
                    "title": f"High Missingness in Column '{col_name}' ({miss_pct:.1f}%)",
                    "description": (
                        f"Column '{col_name}' is missing {miss_count:,} entries out of {len(df):,} "
                        f"total records ({miss_pct:.1f}% missing)."
                    ),
                    "evidence": {
                        "Target Column": col_name,
                        "Missing Values": f"{miss_count:,}",
                        "Missing Rate": f"{miss_pct:.1f}%",
                        "Completeness": f"{100.0 - miss_pct:.1f}%",
                    },
                    "why_it_matters": (
                        "Columns with high missingness may bias analytical metrics or cause errors in "
                        "downstream calculations if nulls are not handled or imputed properly."
                    ),
                }
            )

    # 3. Overall dataset integrity praise if clean
    if dup_count == 0 and summary["missing_cells"] == 0:
        insights.append(
            {
                "id": "qual_clean_dataset",
                "category": "Data Quality",
                "severity": "Positive",
                "title": "High Data Quality & Integrity",
                "description": (
                    f"The dataset achieves 100% completeness with zero missing values and zero duplicate records "
                    f"across all {summary['total_rows']:,} rows and {summary['total_columns']} dimensions."
                ),
                "evidence": {
                    "Completeness": "100.0%",
                    "Missing Cells": "0",
                    "Duplicate Rows": "0",
                },
                "why_it_matters": (
                    "Clean, fully populated data ensures that subsequent statistical aggregates and "
                    "decisions are reliable and free from imputation distortion."
                ),
            }
        )

    return insights


def generate_numerical_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify statistical anomalies, outliers, and extreme dispersion."""
    if df.empty or len(df) < 4:
        return []

    insights: list[dict[str, Any]] = []
    stats_df = compute_numerical_statistics(df)
    if stats_df.empty:
        return []

    for col in stats_df.index:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue

        q1 = stats_df.loc[col, "Q1 (25%)"]
        q3 = stats_df.loc[col, "Q3 (75%)"]
        iqr = stats_df.loc[col, "IQR"]
        mean_val = stats_df.loc[col, "Mean"]
        median_val = stats_df.loc[col, "Median"]
        std_val = stats_df.loc[col, "Std Dev"]

        # A. Outlier detection using IQR rule
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = len(outliers)

            if outlier_count > 0:
                outlier_pct = (outlier_count / len(series)) * 100.0
                insights.append(
                    {
                        "id": f"num_outliers_{col}",
                        "category": "Numerical",
                        "severity": "Warning" if outlier_pct > 3.0 else "Info",
                        "title": f"Statistical Outliers in '{col}' ({outlier_count:,} values)",
                        "description": (
                            f"Detected {outlier_count:,} outlier records ({outlier_pct:.1f}% of values) "
                            f"falling outside standard IQR boundaries [{lower_bound:,.2f}, {upper_bound:,.2f}]."
                        ),
                        "evidence": {
                            "Feature": str(col),
                            "Outlier Count": f"{outlier_count:,}",
                            "Outlier Rate": f"{outlier_pct:.1f}%",
                            "Upper Bound": f"{upper_bound:,.2f}",
                            "Observed Max": f"{series.max():,.2f}",
                        },
                        "why_it_matters": (
                            "Outliers can disproportionately drag arithmetic averages. "
                            "Consider examining these records to determine if they represent data entry errors or high-value business events."
                        ),
                    }
                )

        # B. High coefficient of variation (CV = std / mean) for positive scales
        if mean_val > 0 and std_val > 0:
            cv = std_val / mean_val
            if cv >= 1.2:
                insights.append(
                    {
                        "id": f"num_dispersion_{col}",
                        "category": "Numerical",
                        "severity": "Info",
                        "title": f"High Variability in '{col}' (CV = {cv:.2f})",
                        "description": (
                            f"Feature '{col}' exhibits significant dispersion relative to its mean "
                            f"(Standard Deviation: {std_val:,.2f} vs. Mean: {mean_val:,.2f})."
                        ),
                        "evidence": {
                            "Feature": str(col),
                            "Mean": f"{mean_val:,.2f}",
                            "Median": f"{median_val:,.2f}",
                            "Std Dev": f"{std_val:,.2f}",
                            "Coefficient of Var": f"{cv:.2f}",
                        },
                        "why_it_matters": (
                            "When variability is very high, the median is often a more reliable representative measure "
                            "than the mean for business decision-making."
                        ),
                    }
                )

    return insights


def generate_categorical_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect dominant categories, severe class imbalance, and rare segments."""
    if df.empty or len(df) < 5:
        return []

    insights: list[dict[str, Any]] = []
    classified = classify_columns(df)
    cat_cols = classified["categorical"]

    for col in cat_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        total_vals = len(series)
        counts = series.value_counts()
        if counts.empty:
            continue

        top_cat = counts.index[0]
        top_count = int(counts.iloc[0])
        top_pct = (top_count / total_vals) * 100.0

        # Dominance threshold (>= 60% of entries in one category)
        if top_pct >= 60.0 and len(counts) > 1:
            insights.append(
                {
                    "id": f"cat_dominant_{col}",
                    "category": "Categorical",
                    "severity": "Info",
                    "title": f"Category Concentration in '{col}': '{top_cat}' ({top_pct:.1f}%)",
                    "description": (
                        f"The category '{top_cat}' accounts for {top_count:,} out of {total_vals:,} "
                        f"records ({top_pct:.1f}% concentration)."
                    ),
                    "evidence": {
                        "Feature": str(col),
                        "Dominant Segment": str(top_cat),
                        "Volume": f"{top_count:,}",
                        "Concentration": f"{top_pct:.1f}%",
                        "Other Segments": f"{len(counts) - 1:,}",
                    },
                    "why_it_matters": (
                        "High concentration indicates that business performance is heavily dependent "
                        "on a single category, which may represent either a core strength or a diversification risk."
                    ),
                }
            )

        # Rare categories check (categories appearing in <= 1% of rows when multiple categories exist)
        if len(counts) >= 4 and total_vals >= 50:
            rare_cats = counts[counts / total_vals < 0.02]
            if len(rare_cats) >= 2:
                sample_rare = ", ".join(list(rare_cats.index[:3]))
                insights.append(
                    {
                        "id": f"cat_rare_{col}",
                        "category": "Categorical",
                        "severity": "Info",
                        "title": f"Fragmented Long Tail in '{col}' ({len(rare_cats)} rare categories)",
                        "description": (
                            f"Column '{col}' contains {len(rare_cats)} categories each representing under 2% "
                            f"of total volume (e.g. {sample_rare})."
                        ),
                        "evidence": {
                            "Feature": str(col),
                            "Rare Categories": f"{len(rare_cats)}",
                            "Total Categories": f"{len(counts)}",
                        },
                        "why_it_matters": (
                            "Numerous rare categories may indicate unstandardized data entry or opportunities "
                            "to consolidate low-volume segments into an 'Other' grouping."
                        ),
                    }
                )

    return insights


def generate_correlation_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify strong statistical correlations between numerical features."""
    if df.empty or len(df) < 4:
        return []

    insights: list[dict[str, Any]] = []
    corr_matrix, _ = compute_correlation_matrix(df)
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return []

    top_pairs = get_top_correlations(corr_matrix, top_n=5)

    for pair in top_pairs:
        r_val = pair["correlation"]
        abs_r = pair["abs_correlation"]
        feature_a = pair["feature_a"]
        feature_b = pair["feature_b"]

        if abs_r >= 0.65:
            direction = "positive" if r_val > 0 else "negative"
            insights.append(
                {
                    "id": f"corr_{feature_a}_{feature_b}",
                    "category": "Correlations",
                    "severity": "Info",
                    "title": f"Strong {direction.capitalize()} Correlation: '{feature_a}' & '{feature_b}'",
                    "description": (
                        f"'{feature_a}' and '{feature_b}' have a Pearson correlation coefficient of {r_val:+.2f}, "
                        f"indicating a strong {direction} linear co-movement."
                    ),
                    "evidence": {
                        "Feature 1": str(feature_a),
                        "Feature 2": str(feature_b),
                        "Pearson r": f"{r_val:+.2f}",
                        "Strength": pair["relationship"],
                    },
                    "why_it_matters": (
                        f"When '{feature_a}' {'increases' if r_val > 0 else 'increases'}, "
                        f"'{feature_b}' tends to {'increase' if r_val > 0 else 'decrease'}. "
                        "Note: Statistical correlation demonstrates co-movement, not direct cause-and-effect."
                    ),
                }
            )

    return insights


def generate_time_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect time-series trajectories, changes, and historical span."""
    if df.empty or len(df) < 5:
        return []

    insights: list[dict[str, Any]] = []
    classified = classify_columns(df)
    num_cols = classified["numerical"]

    # Locate datetime column
    date_candidates = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in str(c).lower() or "time" in str(c).lower()
    ]

    if not date_candidates or not num_cols:
        return []

    date_col = date_candidates[0]
    metric_col = num_cols[0]

    # Parse and clean time series
    ts_df = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()

    if len(ts_df) < 4:
        return []

    ts_df = ts_df.sort_values(by="date")
    start_date = ts_df["date"].min().strftime("%Y-%m-%d")
    end_date = ts_df["date"].max().strftime("%Y-%m-%d")
    days_span = (ts_df["date"].max() - ts_df["date"].min()).days

    # Split into first half vs second half to determine broad trajectory
    midpoint = len(ts_df) // 2
    first_half = ts_df.iloc[:midpoint]["metric"].mean()
    second_half = ts_df.iloc[midpoint:]["metric"].mean()

    if first_half > 0:
        pct_change = ((second_half - first_half) / first_half) * 100.0
        direction = "Upward" if pct_change > 5.0 else ("Downward" if pct_change < -5.0 else "Stable")

        severity = "Positive" if pct_change > 15.0 else ("Warning" if pct_change < -15.0 else "Info")
        insights.append(
            {
                "id": f"time_trend_{metric_col}",
                "category": "Trends",
                "severity": severity,
                "title": f"{direction} Trend in '{metric_col}' ({start_date} to {end_date})",
                "description": (
                    f"Comparing the initial period to the later period across a {days_span:,}-day span, "
                    f"average '{metric_col}' shifted by {pct_change:+.1f}% "
                    f"(from {first_half:,.2f} to {second_half:,.2f})."
                ),
                "evidence": {
                    "Timeline": f"{start_date} to {end_date}",
                    "Span": f"{days_span:,} days",
                    "Baseline Average": f"{first_half:,.2f}",
                    "Recent Average": f"{second_half:,.2f}",
                    "Shift Rate": f"{pct_change:+.1f}%",
                },
                "why_it_matters": (
                    f"Understanding whether '{metric_col}' is growing or contracting over time provides "
                    "vital directional guidance for capacity planning and operational forecasts."
                ),
            }
        )

    return insights


def generate_business_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Derive transactional, revenue, and status-driven business takeaways."""
    if df.empty:
        return []

    insights: list[dict[str, Any]] = []
    kpi_results = compute_business_kpis(df)
    inferred = kpi_results["inferred_kpis"]

    # 1. Inferred Revenue & Transaction Volume
    rev_kpis = [k for k in inferred if "Revenue" in k["name"]]
    if rev_kpis:
        rev_kpi = rev_kpis[0]
        aov_kpis = [k for k in inferred if "Average" in k["name"]]
        aov_str = aov_kpis[0]["formatted"] if aov_kpis else "N/A"

        insights.append(
            {
                "id": "biz_inferred_revenue",
                "category": "Business",
                "severity": "Positive",
                "title": f"Inferred Revenue Throughput: {rev_kpi['formatted']}",
                "description": (
                    f"Calculated {rev_kpi['formatted']} in gross revenue across {len(df):,} transactions, "
                    f"with an average order/transaction value of {aov_str}."
                ),
                "evidence": {
                    "Total Revenue": rev_kpi["formatted"],
                    "Average Transaction": aov_str,
                    "Total Volume": f"{len(df):,} records",
                },
                "why_it_matters": (
                    f"Establishes a quantitative revenue baseline. {rev_kpi['assumption']}"
                ),
            }
        )

    # 2. Status Distribution Insights (e.g. cancellation or delivery rates)
    status_df = kpi_results["status_distribution"]
    if status_df is not None and not status_df.empty:
        # Check for friction / cancellation statuses
        negative_statuses = status_df[
            status_df["Status"].str.lower().isin(["cancelled", "canceled", "failed", "returned", "refunded"])
        ]
        if not negative_statuses.empty:
            neg_row = negative_statuses.iloc[0]
            neg_name = str(neg_row["Status"])
            neg_count = int(neg_row["Count"])
            neg_pct = float(neg_row["Percentage (%)"])

            severity = "Critical" if neg_pct >= 15.0 else ("Warning" if neg_pct >= 5.0 else "Info")
            insights.append(
                {
                    "id": f"biz_status_negative_{neg_name}",
                    "category": "Business",
                    "severity": severity,
                    "title": f"Order Friction: '{neg_name}' Rate at {neg_pct:.1f}%",
                    "description": (
                        f"Detected {neg_count:,} orders in '{neg_name}' status, representing {neg_pct:.1f}% "
                        f"of total transactional flow."
                    ),
                    "evidence": {
                        "Status Label": neg_name,
                        "Incident Count": f"{neg_count:,}",
                        "Incident Rate": f"{neg_pct:.1f}%",
                    },
                    "why_it_matters": (
                        "Elevated cancellation, return, or failure rates directly impact net revenue and "
                        "signal customer experience or fulfillment bottlenecks."
                    ),
                }
            )

        # Check for dominant fulfillment status
        top_status = status_df.iloc[0]
        top_name = str(top_status["Status"])
        top_count = int(top_status["Count"])
        top_pct = float(top_status["Percentage (%)"])

        if top_pct >= 70.0:
            insights.append(
                {
                    "id": f"biz_status_dominant_{top_name}",
                    "category": "Business",
                    "severity": "Positive",
                    "title": f"High Fulfillment Stability: '{top_name}' at {top_pct:.1f}%",
                    "description": (
                        f"{top_pct:.1f}% of all tracked records ({top_count:,} out of {len(df):,}) "
                        f"are in '{top_name}' status."
                    ),
                    "evidence": {
                        "Leading Status": top_name,
                        "Record Count": f"{top_count:,}",
                        "Proportion": f"{top_pct:.1f}%",
                    },
                    "why_it_matters": (
                        "Indicates operational consistency and high completion rates across the workflow."
                    ),
                }
            )

    return insights


def generate_smart_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Synthesize all modular insights for the active dataset.

    Returns:
        List of distinct, verified insight dictionaries sorted by severity.
    """
    if df.empty:
        return []

    all_insights: list[dict[str, Any]] = []

    # Run individual analyzers
    all_insights.extend(generate_quality_insights(df))
    all_insights.extend(generate_business_insights(df))
    all_insights.extend(generate_numerical_insights(df))
    all_insights.extend(generate_categorical_insights(df))
    all_insights.extend(generate_correlation_insights(df))
    all_insights.extend(generate_time_insights(df))

    # Priority sorting order: Critical -> Warning -> Info -> Positive
    severity_order = {"Critical": 0, "Warning": 1, "Info": 2, "Positive": 3}
    all_insights.sort(key=lambda x: severity_order.get(x.get("severity", "Info"), 99))

    return all_insights
