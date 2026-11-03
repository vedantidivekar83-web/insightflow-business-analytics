"""Plotly visualization factory for InsightFlow.

Generates publication-quality, theme-consistent interactive charts
for distributions, categorical frequencies, correlations, and time-series trends.
Styled for the 'Futuristic Purple Glass Analytics Studio'.
"""

from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Theme constants for Futuristic Purple Glass Analytics Studio
CHART_FONT_FAMILY: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
COLOR_PRIMARY: str = "#8B5CF6"    # Primary Violet
COLOR_SECONDARY: str = "#A855F7"  # Electric Purple
COLOR_CYAN: str = "#22D3EE"       # Cyan
COLOR_PINK: str = "#EC4899"       # Pink
COLOR_MINT: str = "#34D399"       # Mint
COLOR_AMBER: str = "#FBBF24"      # Amber
COLOR_ROSE: str = "#FB7185"       # Rose
COLOR_GRID: str = "rgba(167, 139, 250, 0.12)"   # Subtle violet grid
COLOR_BORDER: str = "rgba(167, 139, 250, 0.22)" # Illuminated border
COLOR_BG: str = "#17172F"         # Glass panel background
COLOR_PLOT_BG: str = "#101024"    # Inner canvas background


def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent Futuristic Purple Glass typography and theme settings to a Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": CHART_FONT_FAMILY, "size": 14, "color": "#F8FAFC"},
            "x": 0.0,
            "xanchor": "left",
        },
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PLOT_BG,
        font={"family": CHART_FONT_FAMILY, "color": "#A5A5C5"},
        margin=dict(l=45, r=25, t=48, b=42),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(family=CHART_FONT_FAMILY, size=11, color="#A5A5C5"),
            title_font=dict(family=CHART_FONT_FAMILY, size=12, color="#F8FAFC"),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(family=CHART_FONT_FAMILY, size=11, color="#A5A5C5"),
            title_font=dict(family=CHART_FONT_FAMILY, size=12, color="#F8FAFC"),
        ),
        hoverlabel=dict(
            bgcolor="#20203D",
            bordercolor=COLOR_PRIMARY,
            font_family=CHART_FONT_FAMILY,
            font_size=12,
            font_color="#F8FAFC",
        ),
        legend=dict(
            font=dict(family=CHART_FONT_FAMILY, size=11, color="#A5A5C5"),
            bgcolor="rgba(23, 23, 47, 0.75)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
        ),
    )
    return fig


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Generate a distribution histogram with mean and median indicators."""
    if df.empty or column not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Data Available")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        fig = go.Figure()
        return _apply_theme(fig, f"Distribution of {column} (No Numeric Data)")

    mean_val = float(series.mean())
    median_val = float(series.median())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=30,
            marker=dict(
                color=COLOR_PRIMARY,
                line=dict(color="#C4B5FD", width=1),
            ),
            opacity=0.85,
            name=column,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    # Reference lines for Mean and Median
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLOR_PINK,
        annotation_text=f"Mean: {mean_val:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_PINK, size=11, family=CHART_FONT_FAMILY),
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=COLOR_CYAN,
        annotation_text=f"Median: {median_val:,.2f}",
        annotation_position="bottom right",
        annotation_font=dict(color=COLOR_CYAN, size=11, family=CHART_FONT_FAMILY),
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Record Count",
        bargap=0.06,
    )
    return _apply_theme(fig, f"Distribution Analysis: {column}")


def plot_categorical_frequency(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> go.Figure:
    """Generate a clean horizontal bar chart displaying top categories."""
    if df.empty or column not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Data Available")

    counts = df[column].dropna().astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]

    if counts.empty:
        fig = go.Figure()
        return _apply_theme(fig, f"Frequency of {column} (Empty)")

    # Invert so highest is at the top
    counts = counts.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=counts["Count"],
            y=counts[column],
            orientation="h",
            marker=dict(
                color=counts["Count"],
                colorscale=[[0, "#2E1065"], [0.5, "#8B5CF6"], [1.0, "#22D3EE"]],
                showscale=False,
                line=dict(color="#A855F7", width=1),
            ),
            text=counts["Count"],
            textposition="auto",
            textfont=dict(color="#F8FAFC", family=CHART_FONT_FAMILY),
            hovertemplate="Category: %{y}<br>Count: %{x}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Count",
        yaxis_title=column,
    )
    return _apply_theme(fig, f"Top {len(counts)} Categories: {column}")


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Generate an annotated correlation heatmap in Futuristic Purple Glass theme."""
    if corr_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "Correlation Matrix (Insufficient Numerical Data)")

    cols = list(corr_df.columns)
    z_values = corr_df.values

    # Format text labels for cells
    text_values = [[f"{val:.2f}" for val in row] for row in z_values]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=cols,
            y=cols,
            colorscale=[[0, "#EC4899"], [0.5, "#101024"], [1.0, "#22D3EE"]],
            zmin=-1.0,
            zmax=1.0,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11, "family": CHART_FONT_FAMILY, "color": "#F8FAFC"},
            colorbar=dict(
                title=dict(text="Pearson r", font=dict(color="#F8FAFC", family=CHART_FONT_FAMILY)),
                thickness=14,
                tickfont=dict(color="#A5A5C5", family=CHART_FONT_FAMILY),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        xaxis=dict(tickangle=-35),
        yaxis=dict(autorange="reversed"),
    )
    return _apply_theme(fig, "Pearson Correlation Heatmap")


def plot_time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
) -> go.Figure:
    """Generate an interactive time-series line trend chart."""
    if df.empty or date_col not in df.columns or metric_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Time-Series Trend (Data Unavailable)")

    # Prepare datetime index and clean numerical metric
    temp_df = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()

    if temp_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, f"Trend: {metric_col} over {date_col} (No Valid Dates)")

    # Group by date (daily aggregate)
    trend_series = temp_df.groupby("date")["metric"].sum().reset_index()
    trend_series = trend_series.sort_values(by="date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_series["date"],
            y=trend_series["metric"],
            mode="lines+markers",
            line=dict(color=COLOR_CYAN, width=2.5),
            marker=dict(size=6, color=COLOR_SECONDARY, line=dict(color="#FFFFFF", width=1)),
            name=metric_col,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=f"Total {metric_col}",
    )
    return _apply_theme(fig, f"Trend Over Time: Total {metric_col} by {date_col}")
