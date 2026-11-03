"""Plotly visualization components for InsightFlow Machine Learning & AI insights.

Provides interactive charts styled for the 'Futuristic Purple Glass Analytics Studio':
- Multivariate anomaly detection scatter plots
- Feature importance ranking horizontal bar charts
- Actual vs Predicted regression alignment plots
- Classification confusion matrix heatmaps
- Time-series forecasting charts with expanding uncertainty confidence ribbons
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
    COLOR_CYAN,
    COLOR_PINK,
    COLOR_MINT,
    COLOR_ROSE,
    COLOR_GRID,
    COLOR_BG,
    COLOR_PLOT_BG,
)


def plot_anomaly_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    anomaly_col: str = "Anomaly Severity (%)",
) -> go.Figure:
    """Generate a 2D scatter plot highlighting anomalies against normal observations."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Anomaly Scatter Plot (No Data Available)")

    fig = go.Figure()

    is_anomaly = df[anomaly_col] > 50.0 if anomaly_col in df.columns else pd.Series(False, index=df.index)
    normal_df = df[~is_anomaly]
    anom_df = df[is_anomaly]

    # Normal observations in Cyan
    fig.add_trace(
        go.Scatter(
            x=normal_df[x_col],
            y=normal_df[y_col],
            mode="markers",
            name="Normal Record",
            marker=dict(color=COLOR_CYAN, size=6, opacity=0.75),
            hovertemplate=f"<b>{x_col}:</b> %{{x:,.2f}}<br><b>{y_col}:</b> %{{y:,.2f}}<br>Status: Normal<extra></extra>",
        )
    )

    # Flagged Anomalies in Electric Pink
    if not anom_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anom_df[x_col],
                y=anom_df[y_col],
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(color=COLOR_PINK, size=9, symbol="diamond", line=dict(color="#FCE7F3", width=1.5)),
                hovertemplate=f"<b>{x_col}:</b> %{{x:,.2f}}<br><b>{y_col}:</b> %{{y:,.2f}}<br><b>Status: Anomaly</b><extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_theme(fig, f"Isolation Forest Anomaly Isolation: {y_col} vs {x_col}")


def plot_feature_importance(imp_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Generate a horizontal bar chart displaying top predictive feature importances."""
    if imp_df.empty or "Feature" not in imp_df.columns or "Importance (%)" not in imp_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Feature Importance (No Data Available)")

    chart_df = imp_df.head(top_n).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=chart_df["Importance (%)"],
            y=chart_df["Feature"],
            orientation="h",
            marker=dict(
                color=chart_df["Importance (%)"],
                colorscale=[[0, "#2E1065"], [0.5, "#8B5CF6"], [1.0, "#22D3EE"]],
                showscale=False,
                line=dict(color="#A855F7", width=1),
            ),
            text=[f"{v:.1f}%" for v in chart_df["Importance (%)"]],
            textposition="auto",
            textfont=dict(color="#F8FAFC", family=CHART_FONT_FAMILY),
            hovertemplate="<b>Feature:</b> %{y}<br><b>Importance:</b> %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Relative Importance (%)",
        yaxis_title="Predictor Feature",
    )
    return _apply_theme(fig, f"Top {len(chart_df)} Predictive Feature Importances")


def plot_actual_vs_predicted(
    sample_df: pd.DataFrame,
    actual_col: str = "Actual Value",
    pred_col: str = "Model Prediction",
) -> go.Figure:
    """Generate a regression alignment plot comparing actual vs predicted test values."""
    if sample_df.empty or actual_col not in sample_df.columns or pred_col not in sample_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Actual vs Predicted (No Data Available)")

    fig = go.Figure()

    # Actual vs Predicted points in Cyan
    fig.add_trace(
        go.Scatter(
            x=sample_df[actual_col],
            y=sample_df[pred_col],
            mode="markers",
            name="Test Observations",
            marker=dict(color=COLOR_CYAN, size=8, opacity=0.8, line=dict(color="#67E8F9", width=1)),
            hovertemplate=f"<b>Actual:</b> %{{x:,.2f}}<br><b>Predicted:</b> %{{y:,.2f}}<extra></extra>",
        )
    )

    # 45-degree perfect prediction reference line
    min_val = min(float(sample_df[actual_col].min()), float(sample_df[pred_col].min()))
    max_val = max(float(sample_df[actual_col].max()), float(sample_df[pred_col].max()))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction Line",
            line=dict(color="#A5A5C5", dash="dash", width=2),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title=actual_col,
        yaxis_title=pred_col,
    )
    return _apply_theme(fig, "Model Alignment: Actual vs Predicted (Test Set)")


def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame) -> go.Figure:
    """Generate an annotated confusion matrix heatmap for classification evaluation."""
    if cm_df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "Confusion Matrix (Empty Data)")

    z_vals = cm_df.values
    classes = [str(c) for c in cm_df.columns]
    text_labels = [[f"{val:,}" for val in row] for row in z_vals]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=classes,
            y=classes,
            colorscale=[[0, "#101024"], [0.5, "#6D28D9"], [1.0, "#22D3EE"]],
            text=text_labels,
            texttemplate="%{text}",
            textfont=dict(size=13, family=CHART_FONT_FAMILY, color="#F8FAFC"),
            colorbar=dict(
                title=dict(text="Records", font=dict(color="#F8FAFC", family=CHART_FONT_FAMILY)),
                thickness=14,
                tickfont=dict(color="#A5A5C5", family=CHART_FONT_FAMILY),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        xaxis=dict(title="Predicted Class"),
        yaxis=dict(title="Actual Observed Class", autorange="reversed"),
    )
    return _apply_theme(fig, "Classification Confusion Matrix")


def plot_forecast_chart(
    combined_df: pd.DataFrame,
    metric_label: str = "Metric",
) -> go.Figure:
    """Generate an interactive time-series forecast chart with historical data and confidence ribbon."""
    if combined_df.empty or "period" not in combined_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "Time-Series Forecast (No Data Available)")

    hist_df = combined_df[combined_df["type"] == "Historical"]
    future_df = combined_df[combined_df["type"] == "Forecast"]

    fig = go.Figure()

    # 1. Uncertainty Ribbon in soft purple glow
    if not future_df.empty and "upper_bound" in future_df.columns and "lower_bound" in future_df.columns:
        x_future = list(future_df["period"])
        y_upper = list(future_df["upper_bound"])
        y_lower = list(future_df["lower_bound"])

        # Connect ribbon to last historical point
        if not hist_df.empty:
            last_p = hist_df.iloc[-1]["period"]
            last_v = hist_df.iloc[-1]["actual"]
            x_future = [last_p] + x_future
            y_upper = [last_v] + y_upper
            y_lower = [last_v] + y_lower

        fig.add_trace(
            go.Scatter(
                x=x_future + x_future[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor="rgba(139, 92, 246, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Band",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # 2. Historical Actual Observations in Cyan
    if not hist_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_df["period"],
                y=hist_df["actual"],
                mode="lines+markers",
                name="Historical Actual",
                line=dict(color=COLOR_CYAN, width=2.5),
                marker=dict(size=6, color=COLOR_CYAN),
                hovertemplate="<b>Period:</b> %{x}<br><b>Actual:</b> %{y:,.2f}<extra></extra>",
            )
        )

    # 3. Model Fitted Trajectory (historical in-sample fit)
    if not hist_df.empty and "fitted" in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_df["period"],
                y=hist_df["fitted"],
                mode="lines",
                name="Fitted Model",
                line=dict(color="#A5A5C5", dash="dot", width=1.8),
                hoverinfo="skip",
            )
        )

    # 4. Projected Future Forecast in Electric Purple / Pink
    if not future_df.empty:
        x_forecast = list(future_df["period"])
        y_forecast = list(future_df["forecast"])
        if not hist_df.empty:
            x_forecast = [hist_df.iloc[-1]["period"]] + x_forecast
            y_forecast = [hist_df.iloc[-1]["actual"]] + y_forecast

        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=y_forecast,
                mode="lines+markers",
                name="Projected Forecast",
                line=dict(color=COLOR_PINK, width=2.5, dash="dash"),
                marker=dict(size=7, color=COLOR_PINK),
                hovertemplate="<b>Period:</b> %{x}<br><b>Forecast:</b> %{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Timeline Period",
        yaxis_title=metric_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return _apply_theme(fig, f"Time-Series Horizon: Historical vs Projected {metric_label}")
