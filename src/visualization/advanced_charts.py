"""Advanced Plotly visualization components for InsightFlow.

Provides publication-ready interactive visualizations:
- Period trend trajectory line charts with growth rates and reference benchmarks
- Comparative grouped/stacked bar charts across categories
- Segmentation donut and tier proportion charts
- Performance ranking bar charts (Top N vs Bottom N comparison)
- Grouped distribution box plots
- Bivariate scatter plots with linear regression trendlines
"""

from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.visualization.charts import (
    _apply_theme,
    CHART_FONT_FAMILY,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_GRID,
    COLOR_BG,
    COLOR_PLOT_BG,
)


def plot_trend_analysis_chart(
    timeline_df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    frequency_label: str = "Monthly",
    average_val: float | None = None,
) -> go.Figure:
    """Generate an interactive time-series trend line chart with period growth tooltips.

    Args:
        timeline_df: DataFrame with 'period', 'value', 'change', 'pct_change'.
        date_col: Name of date dimension.
        metric_col: Name of numerical metric.
        frequency_label: Human-readable frequency label.
        average_val: Optional benchmark mean value to plot as a reference line.

    Returns:
        Plotly Figure.
    """
    if timeline_df.empty or "period" not in timeline_df.columns or "value" not in timeline_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Trend Analysis (No Data Available)")

    fig = go.Figure()

    # Main trend line
    hover_texts = []
    for _, row in timeline_df.iterrows():
        pct_str = f"{row.get('pct_change', 0):+.1f}%" if pd.notna(row.get("pct_change")) else "N/A"
        change_str = f"{row.get('change', 0):+,.2f}" if pd.notna(row.get("change")) else "N/A"
        txt = (
            f"<b>Period:</b> {row['period']}<br>"
            f"<b>{metric_col}:</b> {row['value']:,.2f}<br>"
            f"<b>Period Change:</b> {change_str} ({pct_str})"
        )
        hover_texts.append(txt)

    fig.add_trace(
        go.Scatter(
            x=timeline_df["period"],
            y=timeline_df["value"],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=7, color=COLOR_SECONDARY, symbol="circle"),
            name=f"{metric_col} ({frequency_label})",
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    # Reference benchmark line for period average
    if average_val is not None:
        fig.add_hline(
            y=average_val,
            line_dash="dash",
            line_color="#64748B",
            annotation_text=f"Mean: {average_val:,.2f}",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#64748B"),
        )

    fig.update_layout(
        xaxis_title=f"Timeline ({frequency_label})",
        yaxis_title=metric_col,
        hovermode="closest",
    )
    return _apply_theme(fig, f"Trend Trajectory: {metric_col} ({frequency_label})")


def plot_comparative_chart(
    comp_df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    secondary_col: str | None = None,
    pivot_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Generate a comparative grouped or single bar chart.

    Args:
        comp_df: Summary DataFrame with group_col, 'Total', 'Mean', 'Share (%)'.
        group_col: Category column name.
        metric_col: Metric column name.
        secondary_col: Optional secondary dimension.
        pivot_df: Optional pivot table for cross-tabulated grouped bar chart.

    Returns:
        Plotly Figure.
    """
    if comp_df.empty or group_col not in comp_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Comparative Analysis (No Data Available)")

    # Case 1: Cross-tabulation grouped bar chart
    if secondary_col and pivot_df is not None and not pivot_df.empty:
        fig = go.Figure()
        palette = px.colors.qualitative.Safe

        for idx, sub_cat in enumerate(pivot_df.columns):
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Bar(
                    name=str(sub_cat),
                    x=pivot_df.index.astype(str),
                    y=pivot_df[sub_cat],
                    marker_color=color,
                    hovertemplate=f"<b>{group_col}:</b> %{{x}}<br><b>{secondary_col}:</b> {sub_cat}<br><b>Total:</b> %{{y:,.2f}}<extra></extra>",
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_title=group_col,
            yaxis_title=f"Total {metric_col}",
            legend_title=secondary_col,
        )
        return _apply_theme(fig, f"Comparative Analysis: {metric_col} by {group_col} & {secondary_col}")

    # Case 2: Standard single-dimension bar chart
    chart_df = comp_df.head(15).iloc[::-1]  # Highest at top

    fig = go.Figure(
        go.Bar(
            x=chart_df["Total"],
            y=chart_df[group_col].astype(str),
            orientation="h",
            marker=dict(
                color=chart_df["Total"],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{val:,.2f} ({share:.1f}%)" for val, share in zip(chart_df["Total"], chart_df["Share (%)"])],
            textposition="auto",
            hovertemplate=(
                f"<b>{group_col}:</b> %{{y}}<br>"
                f"<b>Total {metric_col}:</b> %{{x:,.2f}}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title=f"Total {metric_col}",
        yaxis_title=group_col,
    )
    return _apply_theme(fig, f"Comparative Breakdown: {metric_col} by {group_col}")


def plot_segmentation_chart(
    seg_df: pd.DataFrame,
    label_col: str,
    count_col: str = "Records",
    title: str = "Segment Proportion Breakdown",
) -> go.Figure:
    """Generate an interactive donut chart illustrating segment proportions.

    Args:
        seg_df: DataFrame containing segment labels and record counts.
        label_col: Column with segment names.
        count_col: Column with record frequencies.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if seg_df.empty or label_col not in seg_df.columns or count_col not in seg_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Segmentation Analysis (No Data Available)")

    palette = px.colors.qualitative.Prism

    fig = go.Figure(
        data=[
            go.Pie(
                labels=seg_df[label_col].astype(str),
                values=seg_df[count_col],
                hole=0.55,
                textinfo="label+percent",
                marker=dict(colors=palette[: len(seg_df)]),
                hovertemplate="<b>Segment:</b> %{label}<br><b>Records:</b> %{value:,} (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return _apply_theme(fig, title)


def plot_performance_ranking_chart(
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
    group_col: str,
    metric_label: str = "Score",
) -> go.Figure:
    """Generate a split performance ranking chart comparing top and bottom performers.

    Args:
        top_df: Leading categories DataFrame.
        bottom_df: Trailing categories DataFrame.
        group_col: Group column name.
        metric_label: Label for the score metric.

    Returns:
        Plotly Figure.
    """
    if top_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "Performance Rankings (No Data Available)")

    # Combine top and bottom with tags
    plot_items = []
    for _, row in top_df.iloc[::-1].iterrows():
        plot_items.append(
            {
                "Group": f"🏆 #{row['Rank']} {row[group_col]}",
                "Score": row["Score"],
                "Tier": "Top Performer",
                "Color": "#2563EB",
                "Share": row.get("Share (%)", 0.0),
            }
        )

    # Avoid duplicating if top and bottom overlap
    existing_names = set(top_df[group_col])
    distinct_bottom = bottom_df[~bottom_df[group_col].isin(existing_names)].iloc[::-1]

    for _, row in distinct_bottom.iterrows():
        plot_items.append(
            {
                "Group": f"⚠️ #{row['Rank']} {row[group_col]}",
                "Score": row["Score"],
                "Tier": "Lowest Performer",
                "Color": "#EF4444",
                "Share": row.get("Share (%)", 0.0),
            }
        )

    chart_df = pd.DataFrame(plot_items)

    fig = go.Figure(
        go.Bar(
            x=chart_df["Score"],
            y=chart_df["Group"],
            orientation="h",
            marker_color=chart_df["Color"],
            text=[f"{s:,.2f} ({sh:.1f}%)" for s, sh in zip(chart_df["Score"], chart_df["Share"])],
            textposition="auto",
            hovertemplate="<b>Category:</b> %{y}<br><b>Score:</b> %{x:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title=metric_label,
        yaxis_title="Group Rank & Entity",
    )
    return _apply_theme(fig, f"Performance Rankings: Top vs Trailing by {metric_label}")


def plot_grouped_box(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    top_n_groups: int = 8,
) -> go.Figure:
    """Generate comparative box plots illustrating numerical distribution across categories.

    Args:
        df: Input DataFrame.
        cat_col: Categorical column.
        num_col: Numerical column.
        top_n_groups: Max number of top categories to display.

    Returns:
        Plotly Figure.
    """
    if df.empty or cat_col not in df.columns or num_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Distribution Box Plot (No Data Available)")

    clean_df = df[[cat_col, num_col]].dropna().copy()
    clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")
    clean_df = clean_df.dropna()

    if clean_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "Distribution Box Plot (Empty)")

    # Filter to top N groups by volume
    top_groups = clean_df[cat_col].value_counts().head(top_n_groups).index.tolist()
    sub = clean_df[clean_df[cat_col].isin(top_groups)].copy()

    fig = go.Figure()
    palette = px.colors.qualitative.Safe

    for i, grp in enumerate(top_groups):
        grp_data = sub[sub[cat_col] == grp][num_col]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Box(
                y=grp_data,
                name=str(grp),
                boxpoints="outliers",
                marker_color=color,
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        xaxis_title=cat_col,
        yaxis_title=num_col,
        showlegend=False,
    )
    return _apply_theme(fig, f"Distribution Comparison: {num_col} by {cat_col}")


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    trend_metrics: dict[str, Any] | None = None,
) -> go.Figure:
    """Generate an interactive bivariate scatter plot with an optional linear regression line.

    Args:
        df: Input DataFrame.
        x_col: Horizontal axis numerical feature.
        y_col: Vertical axis numerical feature.
        color_col: Optional categorical column for color encoding.
        trend_metrics: Optional dict with 'slope', 'intercept', 'pearson_r'.

    Returns:
        Plotly Figure.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Bivariate Scatter Plot (No Data Available)")

    cols = [x_col, y_col]
    if color_col and color_col in df.columns:
        cols.append(color_col)

    clean_df = df[cols].dropna().copy()
    clean_df[x_col] = pd.to_numeric(clean_df[x_col], errors="coerce")
    clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[x_col, y_col])

    if clean_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "Bivariate Scatter Plot (Empty Data)")

    fig = go.Figure()

    # Data points
    if color_col and color_col in clean_df.columns:
        categories = clean_df[color_col].unique()[:8]
        palette = px.colors.qualitative.Safe
        for idx, cat in enumerate(categories):
            cat_sub = clean_df[clean_df[color_col] == cat]
            fig.add_trace(
                go.Scatter(
                    x=cat_sub[x_col],
                    y=cat_sub[y_col],
                    mode="markers",
                    name=str(cat),
                    marker=dict(size=7, color=palette[idx % len(palette)], opacity=0.8),
                    hovertemplate=f"<b>{x_col}:</b> %{{x:,.2f}}<br><b>{y_col}:</b> %{{y:,.2f}}<br><b>{color_col}:</b> {cat}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=clean_df[x_col],
                y=clean_df[y_col],
                mode="markers",
                name="Observations",
                marker=dict(size=7, color=COLOR_PRIMARY, opacity=0.75),
                hovertemplate=f"<b>{x_col}:</b> %{{x:,.2f}}<br><b>{y_col}:</b> %{{y:,.2f}}<extra></extra>",
            )
        )

    # Linear Regression Line overlay
    if trend_metrics and "slope" in trend_metrics and "intercept" in trend_metrics:
        slope = trend_metrics["slope"]
        intercept = trend_metrics["intercept"]
        x_min = float(clean_df[x_col].min())
        x_max = float(clean_df[x_col].max())

        x_line = [x_min, x_max]
        y_line = [(slope * x) + intercept for x in x_line]

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear Fit",
                line=dict(color="#DC2626", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    r_str = f" (r = {trend_metrics['pearson_r']:+.2f})" if trend_metrics and "pearson_r" in trend_metrics else ""
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return _apply_theme(fig, f"Bivariate Relationship: {y_col} vs {x_col}{r_str}")
