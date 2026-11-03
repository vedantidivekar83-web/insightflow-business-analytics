"""Advanced Analytics Engine for InsightFlow.

Provides high-level business analytics routines:
- Trend Analysis (period grouping, growth rate, trajectory classification)
- Comparative Analysis (group breakdowns, secondary pivots, share calculations)
- Segmentation Analysis (categorical dimensions and numerical quantile/range tiers)
- Performance Rankings (top/bottom performers, tie handling, explanations)
- Distribution & Bivariate Relationship Analysis (correlations, linear regression)
"""

from typing import Any, Literal
import pandas as pd
import numpy as np

from src.data.explorer import classify_columns


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns in DataFrame that contain date or timestamp values.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names suitable for time-series analysis.
    """
    if df.empty:
        return []

    date_cols: list[str] = []

    # 1. Native datetime types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(str(col))

    # 2. String/object columns with date-like names or parseable dates
    date_keywords = ["date", "time", "timestamp", "year", "month", "day", "created", "updated", "period"]
    for col in df.columns:
        col_str = str(col)
        if col_str in date_cols:
            continue

        # Skip numerical columns unless column name explicitly suggests a date (e.g., Year, Date)
        if pd.api.types.is_numeric_dtype(df[col]):
            name_lower = col_str.lower()
            if any(k in name_lower for k in ["year", "date"]):
                # Verify values look like realistic years (e.g. 1950 - 2100)
                valid_num = df[col].dropna()
                if not valid_num.empty and valid_num.between(1900, 2100).mean() > 0.8:
                    date_cols.append(col_str)
            continue

        # For object/string columns
        name_lower = col_str.lower()
        has_keyword = any(k in name_lower for k in date_keywords)

        series = df[col].dropna()
        if series.empty:
            continue

        sample = series.head(50)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        valid_fraction = parsed.notna().mean()

        if valid_fraction >= 0.7 or (has_keyword and valid_fraction >= 0.4):
            date_cols.append(col_str)

    return date_cols


def compute_trend_analysis(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    frequency: Literal["D", "W", "M", "Y"] = "M",
    agg_func: Literal["sum", "mean", "count"] = "sum",
) -> dict[str, Any]:
    """Calculate period-over-period trend analysis with growth rates and trajectory classification.

    Args:
        df: Input DataFrame (will not be mutated).
        date_col: Column containing dates.
        metric_col: Column containing numerical metrics.
        frequency: Aggregation frequency ('D'=Daily, 'W'=Weekly, 'M'=Monthly, 'Y'=Yearly).
        agg_func: Aggregation function ('sum', 'mean', 'count').

    Returns:
        Structured dictionary containing timeline DataFrame and summary metadata.
    """
    if df.empty or date_col not in df.columns or metric_col not in df.columns:
        return {
            "status": "error",
            "message": "Dataset is empty or specified columns are missing.",
            "timeline_df": pd.DataFrame(),
            "summary": {},
        }

    # Extract clean subset
    sub = pd.DataFrame(
        {
            "raw_date": pd.to_datetime(df[date_col], errors="coerce"),
            "raw_metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()

    if sub.empty or len(sub) < 2:
        return {
            "status": "error",
            "message": f"Insufficient valid date and numerical observations in '{date_col}' and '{metric_col}'.",
            "timeline_df": pd.DataFrame(),
            "summary": {},
        }

    sub = sub.sort_values(by="raw_date")

    # Map frequency to period format
    freq_labels = {
        "D": "%Y-%m-%d",
        "W": "Week %W (%Y)",
        "M": "%Y-%m",
        "Y": "%Y",
    }
    date_format = freq_labels.get(frequency, "%Y-%m")

    # Create period bucket
    if frequency == "D":
        sub["period"] = sub["raw_date"].dt.strftime("%Y-%m-%d")
        sub["period_start"] = sub["raw_date"].dt.floor("D")
    elif frequency == "W":
        # Group by Monday of the week
        sub["period_start"] = sub["raw_date"].dt.to_period("W").apply(lambda r: r.start_time)
        sub["period"] = sub["period_start"].dt.strftime("%Y-%m-%d")
    elif frequency == "M":
        sub["period"] = sub["raw_date"].dt.strftime("%Y-%m")
        sub["period_start"] = sub["raw_date"].dt.to_period("M").apply(lambda r: r.start_time)
    elif frequency == "Y":
        sub["period"] = sub["raw_date"].dt.strftime("%Y")
        sub["period_start"] = sub["raw_date"].dt.to_period("Y").apply(lambda r: r.start_time)
    else:
        sub["period"] = sub["raw_date"].dt.strftime("%Y-%m")
        sub["period_start"] = sub["raw_date"].dt.floor("D")

    # Aggregate
    if agg_func == "mean":
        agg_series = sub.groupby(["period", "period_start"])["raw_metric"].mean().reset_index()
    elif agg_func == "count":
        agg_series = sub.groupby(["period", "period_start"])["raw_metric"].count().reset_index()
    else:  # default 'sum'
        agg_series = sub.groupby(["period", "period_start"])["raw_metric"].sum().reset_index()

    agg_series = agg_series.sort_values(by="period_start").reset_index(drop=True)
    agg_series = agg_series.rename(columns={"raw_metric": "value"})

    # Calculate period-over-period differences
    agg_series["change"] = agg_series["value"].diff()
    agg_series["pct_change"] = (agg_series["change"] / agg_series["value"].shift(1).replace(0, np.nan)) * 100.0

    period_count = len(agg_series)
    if period_count < 2:
        return {
            "status": "warning",
            "message": "Only a single period was generated; multiple periods are required to calculate trends.",
            "timeline_df": agg_series,
            "summary": {
                "period_count": period_count,
                "trend_direction": "Stable",
                "overall_growth_pct": 0.0,
                "total_change": 0.0,
                "start_period": str(agg_series.iloc[0]["period"]),
                "end_period": str(agg_series.iloc[-1]["period"]),
                "start_value": float(agg_series.iloc[0]["value"]),
                "end_value": float(agg_series.iloc[-1]["value"]),
                "average_value": float(agg_series["value"].mean()),
                "peak_period": str(agg_series.iloc[0]["period"]),
                "peak_value": float(agg_series.iloc[0]["value"]),
                "trough_period": str(agg_series.iloc[0]["period"]),
                "trough_value": float(agg_series.iloc[0]["value"]),
            },
        }

    start_val = float(agg_series.iloc[0]["value"])
    end_val = float(agg_series.iloc[-1]["value"])
    total_change = end_val - start_val

    if start_val != 0:
        overall_growth = ((end_val - start_val) / abs(start_val)) * 100.0
    else:
        overall_growth = None

    # Trend direction based on overall growth and period-to-period trajectory
    if overall_growth is not None:
        if overall_growth > 2.0:
            trend_dir = "Increasing"
        elif overall_growth < -2.0:
            trend_dir = "Decreasing"
        else:
            trend_dir = "Stable"
    else:
        trend_dir = "Increasing" if total_change > 0 else ("Decreasing" if total_change < 0 else "Stable")

    peak_idx = agg_series["value"].idxmax()
    trough_idx = agg_series["value"].idxmin()

    summary = {
        "metric": metric_col,
        "date_column": date_col,
        "frequency": frequency,
        "agg_func": agg_func,
        "period_count": period_count,
        "start_period": str(agg_series.iloc[0]["period"]),
        "end_period": str(agg_series.iloc[-1]["period"]),
        "start_value": round(start_val, 2),
        "end_value": round(end_val, 2),
        "total_change": round(total_change, 2),
        "overall_growth_pct": round(overall_growth, 2) if overall_growth is not None else None,
        "trend_direction": trend_dir,
        "average_value": round(float(agg_series["value"].mean()), 2),
        "peak_period": str(agg_series.loc[peak_idx, "period"]),
        "peak_value": round(float(agg_series.loc[peak_idx, "value"]), 2),
        "trough_period": str(agg_series.loc[trough_idx, "period"]),
        "trough_value": round(float(agg_series.loc[trough_idx, "value"]), 2),
    }

    return {
        "status": "ok",
        "message": "",
        "timeline_df": agg_series,
        "summary": summary,
    }


def compute_comparative_analysis(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    secondary_col: str | None = None,
) -> dict[str, Any]:
    """Compare numerical metric aggregations across categories of group_col.

    Args:
        df: Input DataFrame.
        group_col: Primary categorical column to group by.
        metric_col: Numerical column to aggregate.
        secondary_col: Optional secondary categorical column for cross-comparison.

    Returns:
        Structured dictionary containing summary table, optional pivot table, and top/bottom metadata.
    """
    if df.empty or group_col not in df.columns or metric_col not in df.columns:
        return {
            "status": "error",
            "message": "Dataset is empty or required columns are missing.",
            "summary_table": pd.DataFrame(),
            "pivot_table": pd.DataFrame(),
            "metadata": {},
        }

    # Extract clean data
    cols = [group_col, metric_col]
    if secondary_col and secondary_col in df.columns and secondary_col != group_col:
        cols.append(secondary_col)

    clean_df = df[cols].dropna(subset=[group_col, metric_col]).copy()
    clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[metric_col])

    if clean_df.empty:
        return {
            "status": "error",
            "message": f"No valid numerical values found for '{metric_col}' grouped by '{group_col}'.",
            "summary_table": pd.DataFrame(),
            "pivot_table": pd.DataFrame(),
            "metadata": {},
        }

    # Grouped aggregations
    grouped = clean_df.groupby(group_col)[metric_col].agg(
        Count="count",
        Total="sum",
        Mean="mean",
        Median="median",
        Min="min",
        Max="max",
        Std="std",
    ).reset_index()

    grouped["Std"] = grouped["Std"].fillna(0.0)

    overall_total = float(grouped["Total"].sum())
    if overall_total > 0:
        grouped["Share (%)"] = (grouped["Total"] / overall_total) * 100.0
    else:
        grouped["Share (%)"] = 0.0

    # Sort descending by Total
    grouped = grouped.sort_values(by="Total", ascending=False).reset_index(drop=True)

    top_row = grouped.iloc[0]
    bottom_row = grouped.iloc[-1]

    # Secondary cross-tabulation / pivot if specified
    pivot_df = pd.DataFrame()
    if secondary_col and secondary_col in clean_df.columns:
        pivot_df = clean_df.pivot_table(
            index=group_col,
            columns=secondary_col,
            values=metric_col,
            aggfunc="sum",
            fill_value=0.0,
        )

    metadata = {
        "group_col": group_col,
        "metric_col": metric_col,
        "secondary_col": secondary_col,
        "group_count": len(grouped),
        "overall_total": round(overall_total, 2),
        "overall_mean": round(float(clean_df[metric_col].mean()), 2),
        "top_group": {
            "name": str(top_row[group_col]),
            "total": round(float(top_row["Total"]), 2),
            "share_pct": round(float(top_row["Share (%)"]), 2),
            "mean": round(float(top_row["Mean"]), 2),
        },
        "bottom_group": {
            "name": str(bottom_row[group_col]),
            "total": round(float(bottom_row["Total"]), 2),
            "share_pct": round(float(bottom_row["Share (%)"]), 2),
            "mean": round(float(bottom_row["Mean"]), 2),
        },
    }

    return {
        "status": "ok",
        "message": "",
        "summary_table": grouped,
        "pivot_table": pivot_df,
        "metadata": metadata,
    }


def compute_categorical_segmentation(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str | None = None,
) -> dict[str, Any]:
    """Segment records by a categorical dimension and calculate segment sizes, shares, and metric values.

    Args:
        df: Input DataFrame.
        segment_col: Categorical column to segment by.
        metric_col: Optional numerical metric to aggregate per segment.

    Returns:
        Structured dictionary containing segment breakdown table and dominant/smallest segment metadata.
    """
    if df.empty or segment_col not in df.columns:
        return {
            "status": "error",
            "message": f"Dataset is empty or '{segment_col}' column is missing.",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    sub = df[[segment_col]].dropna().copy()
    if sub.empty:
        return {
            "status": "error",
            "message": f"No valid values in '{segment_col}'.",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    total_records = len(sub)

    if metric_col and metric_col in df.columns:
        sub[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        grouped = sub.groupby(segment_col).agg(
            Records=(segment_col, "count"),
            Total=(metric_col, "sum"),
            Average=(metric_col, "mean"),
        ).reset_index()
    else:
        grouped = sub[segment_col].value_counts().reset_index()
        grouped.columns = [segment_col, "Records"]

    grouped["Share (%)"] = (grouped["Records"] / total_records) * 100.0
    grouped = grouped.sort_values(by="Records", ascending=False).reset_index(drop=True)

    dom = grouped.iloc[0]
    small = grouped.iloc[-1]

    metadata = {
        "segment_col": segment_col,
        "metric_col": metric_col,
        "total_records": total_records,
        "segment_count": len(grouped),
        "dominant_segment": {
            "name": str(dom[segment_col]),
            "records": int(dom["Records"]),
            "share_pct": round(float(dom["Share (%)"]), 2),
        },
        "smallest_segment": {
            "name": str(small[segment_col]),
            "records": int(small["Records"]),
            "share_pct": round(float(small["Share (%)"]), 2),
        },
    }

    return {
        "status": "ok",
        "message": "",
        "segment_table": grouped,
        "metadata": metadata,
    }


def compute_numerical_segmentation(
    df: pd.DataFrame,
    num_col: str,
    method: Literal["quantiles", "ranges"] = "quantiles",
    num_bins: int = 4,
    target_metric: str | None = None,
) -> dict[str, Any]:
    """Segment records into numerical tiers using quantiles or equal-width ranges.

    Args:
        df: Input DataFrame.
        num_col: Numerical column to base tiers on.
        method: 'quantiles' (equal frequency) or 'ranges' (equal width).
        num_bins: Number of segments (2 to 10).
        target_metric: Optional secondary numerical metric to aggregate per tier.

    Returns:
        Structured dictionary containing tier table, dominant tier, and parameters.
    """
    if df.empty or num_col not in df.columns:
        return {
            "status": "error",
            "message": f"Dataset is empty or '{num_col}' column is missing.",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    sub = pd.DataFrame({"value": pd.to_numeric(df[num_col], errors="coerce")}).dropna()
    if sub.empty:
        return {
            "status": "error",
            "message": f"No valid numerical values in '{num_col}'.",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    if target_metric and target_metric in df.columns and target_metric != num_col:
        sub["target"] = pd.to_numeric(df.loc[sub.index, target_metric], errors="coerce")

    # Guard against insufficient distinct values
    unique_vals = sub["value"].nunique()
    if unique_vals < 2:
        return {
            "status": "error",
            "message": f"Column '{num_col}' has only {unique_vals} unique value; segmentation requires at least 2 distinct values.",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    actual_bins = min(num_bins, unique_vals)

    # Binning
    try:
        if method == "quantiles":
            # Quantile labels
            tier_names = ["Tier 1 (Lowest)", "Tier 2 (Moderate)", "Tier 3 (Upper)", "Tier 4 (Top)"]
            if actual_bins == 2:
                labels = ["Low", "High"]
            elif actual_bins == 3:
                labels = ["Low (Tertile 1)", "Medium (Tertile 2)", "High (Tertile 3)"]
            elif actual_bins == 4:
                labels = ["Low (Q1)", "Moderate (Q2)", "Upper (Q3)", "Top (Q4)"]
            elif actual_bins == 5:
                labels = ["Quintile 1", "Quintile 2", "Quintile 3", "Quintile 4", "Quintile 5"]
            else:
                labels = [f"Tier {i+1}" for i in range(actual_bins)]

            sub["segment"], bin_edges = pd.qcut(
                sub["value"],
                q=actual_bins,
                labels=labels,
                retbins=True,
                duplicates="drop",
            )
        else:  # 'ranges'
            sub["segment"], bin_edges = pd.cut(
                sub["value"],
                bins=actual_bins,
                retbins=True,
            )
            # Format interval labels
            sub["segment"] = sub["segment"].astype(str)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to compute numerical bins: {str(e)}",
            "segment_table": pd.DataFrame(),
            "metadata": {},
        }

    total_records = len(sub)

    # Aggregate by segment
    agg_dict: dict[str, Any] = {
        "Records": ("value", "count"),
        "Min": ("value", "min"),
        "Max": ("value", "max"),
        "Mean": ("value", "mean"),
    }
    if "target" in sub.columns:
        agg_dict["Target Mean"] = ("target", "mean")
        agg_dict["Target Total"] = ("target", "sum")

    table = sub.groupby("segment", observed=False).agg(**agg_dict).reset_index()
    table["Share (%)"] = (table["Records"] / total_records) * 100.0

    # Ensure range column string
    table["Range"] = table.apply(lambda r: f"[{r['Min']:,.2f} – {r['Max']:,.2f}]", axis=1)

    dom = table.sort_values(by="Records", ascending=False).iloc[0]
    small = table.sort_values(by="Records", ascending=True).iloc[0]

    metadata = {
        "num_col": num_col,
        "method": method,
        "bins_requested": num_bins,
        "bins_created": len(table),
        "total_records": total_records,
        "target_metric": target_metric,
        "dominant_segment": {
            "name": str(dom["segment"]),
            "records": int(dom["Records"]),
            "share_pct": round(float(dom["Share (%)"]), 2),
            "range": str(dom["Range"]),
        },
        "smallest_segment": {
            "name": str(small["segment"]),
            "records": int(small["Records"]),
            "share_pct": round(float(small["Share (%)"]), 2),
            "range": str(small["Range"]),
        },
    }

    return {
        "status": "ok",
        "message": "",
        "segment_table": table,
        "metadata": metadata,
    }


def compute_performance_rankings(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    agg_func: Literal["sum", "mean"] = "sum",
    top_n: int = 5,
) -> dict[str, Any]:
    """Calculate performance rankings for categorical groups based on a numerical metric.

    Handles ties using standard minimum ranking and provides textual analytical explanations.

    Args:
        df: Input DataFrame.
        group_col: Grouping category column.
        metric_col: Performance metric column.
        agg_func: 'sum' (volume) or 'mean' (intensity).
        top_n: Number of leading and trailing categories to isolate.

    Returns:
        Structured dictionary containing full ranked table, top/bottom performers, and explanation.
    """
    if df.empty or group_col not in df.columns or metric_col not in df.columns:
        return {
            "status": "error",
            "message": "Dataset is empty or required columns are missing.",
            "ranked_table": pd.DataFrame(),
            "top_performers": pd.DataFrame(),
            "bottom_performers": pd.DataFrame(),
            "metadata": {},
        }

    clean_df = df[[group_col, metric_col]].dropna().copy()
    clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors="coerce")
    clean_df = clean_df.dropna()

    if clean_df.empty:
        return {
            "status": "error",
            "message": f"No valid numerical values for '{metric_col}' grouped by '{group_col}'.",
            "ranked_table": pd.DataFrame(),
            "top_performers": pd.DataFrame(),
            "bottom_performers": pd.DataFrame(),
            "metadata": {},
        }

    # Aggregate
    if agg_func == "mean":
        agg_df = clean_df.groupby(group_col)[metric_col].mean().reset_index()
        metric_label = f"Average {metric_col}"
    else:
        agg_df = clean_df.groupby(group_col)[metric_col].sum().reset_index()
        metric_label = f"Total {metric_col}"

    agg_df = agg_df.rename(columns={metric_col: "Score"})

    # Rank with ties handling (method='min' means if tied for 1st, both are 1 and next is 3)
    agg_df["Rank"] = agg_df["Score"].rank(ascending=False, method="min").astype(int)
    agg_df = agg_df.sort_values(by=["Rank", "Score"], ascending=[True, False]).reset_index(drop=True)

    total_score = agg_df["Score"].sum()
    if total_score > 0:
        agg_df["Share (%)"] = (agg_df["Score"] / total_score) * 100.0
        agg_df["Cumulative Share (%)"] = agg_df["Share (%)"].cumsum()
    else:
        agg_df["Share (%)"] = 0.0
        agg_df["Cumulative Share (%)"] = 0.0

    total_groups = len(agg_df)
    n_effective = min(top_n, total_groups)

    top_performers = agg_df.head(n_effective).copy()
    bottom_performers = agg_df.tail(n_effective).iloc[::-1].copy().reset_index(drop=True)

    best = agg_df.iloc[0]
    worst = agg_df.iloc[-1]

    # Narrative explanation
    explanation = (
        f"Evaluated {total_groups:,} distinct categories in '{group_col}' ranked by {agg_func.upper()} of '{metric_col}'. "
        f"Top performer: **{best[group_col]}** with a score of {best['Score']:,.2f} ({best['Share (%)']:.1f}% of total). "
        f"Lowest performer: **{worst[group_col]}** with a score of {worst['Score']:,.2f} ({worst['Share (%)']:.1f}% of total). "
        f"Ranks are assigned descending, where tied values share identical minimum ranks."
    )

    metadata = {
        "group_col": group_col,
        "metric_col": metric_col,
        "metric_label": metric_label,
        "agg_func": agg_func,
        "total_groups": total_groups,
        "top_n_requested": top_n,
        "top_performer": {
            "name": str(best[group_col]),
            "score": round(float(best["Score"]), 2),
            "share_pct": round(float(best["Share (%)"]), 2),
        },
        "bottom_performer": {
            "name": str(worst[group_col]),
            "score": round(float(worst["Score"]), 2),
            "share_pct": round(float(worst["Share (%)"]), 2),
        },
        "explanation": explanation,
    }

    return {
        "status": "ok",
        "message": "",
        "ranked_table": agg_df,
        "top_performers": top_performers,
        "bottom_performers": bottom_performers,
        "metadata": metadata,
    }


def compute_bivariate_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    """Compute bivariate relationship statistics, Pearson correlation, and linear regression parameters.

    Args:
        df: Input DataFrame.
        x_col: Independent feature name.
        y_col: Dependent feature name.

    Returns:
        Structured dictionary containing correlation, R-squared, slope, intercept, and equation.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return {
            "status": "error",
            "message": "Dataset is empty or required columns are missing.",
            "data": pd.DataFrame(),
            "metrics": {},
        }

    sub = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    ).dropna()

    if len(sub) < 3:
        return {
            "status": "error",
            "message": f"Insufficient non-null paired observations between '{x_col}' and '{y_col}' (need at least 3).",
            "data": pd.DataFrame(),
            "metrics": {},
        }

    x_vals = sub["x"].values
    y_vals = sub["y"].values

    x_std = float(np.std(x_vals, ddof=1))
    y_std = float(np.std(y_vals, ddof=1))

    if x_std == 0.0 or y_std == 0.0:
        return {
            "status": "error",
            "message": f"Zero variance detected in '{x_col}' or '{y_col}'; cannot compute correlation or regression.",
            "data": sub,
            "metrics": {},
        }

    # Pearson r
    r = float(np.corrcoef(x_vals, y_vals)[0, 1])
    r_squared = r ** 2

    # Linear Regression (y = mx + b)
    x_mean = float(np.mean(x_vals))
    y_mean = float(np.mean(y_vals))

    covariance = float(np.cov(x_vals, y_vals)[0, 1])
    slope = covariance / (x_std ** 2)
    intercept = y_mean - (slope * x_mean)

    # Classify relationship
    abs_r = abs(r)
    if abs_r >= 0.8:
        strength = "Very Strong"
    elif abs_r >= 0.6:
        strength = "Strong"
    elif abs_r >= 0.4:
        strength = "Moderate"
    elif abs_r >= 0.2:
        strength = "Weak"
    else:
        strength = "Negligible"

    direction = "Positive" if r > 0 else "Negative"
    relationship_label = f"{strength} {direction}" if strength != "Negligible" else "No Linear Correlation"

    sign = "+" if intercept >= 0 else "-"
    equation = f"{y_col} = ({slope:,.4f} * {x_col}) {sign} {abs(intercept):,.2f}"

    metrics = {
        "x_col": x_col,
        "y_col": y_col,
        "sample_size": len(sub),
        "pearson_r": round(r, 4),
        "r_squared": round(r_squared, 4),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "equation": equation,
        "relationship": relationship_label,
        "summary_x": {
            "mean": round(x_mean, 2),
            "std": round(x_std, 2),
            "min": round(float(np.min(x_vals)), 2),
            "max": round(float(np.max(x_vals)), 2),
        },
        "summary_y": {
            "mean": round(y_mean, 2),
            "std": round(y_std, 2),
            "min": round(float(np.min(y_vals)), 2),
            "max": round(float(np.max(y_vals)), 2),
        },
    }

    return {
        "status": "ok",
        "message": "",
        "data": sub,
        "metrics": metrics,
    }
