"""Time-Series Forecasting engine for InsightFlow.

Provides explainable, lightweight statistical baseline forecasting (Holt's Linear Exponential Smoothing)
with prediction uncertainty intervals and historical continuity verification.
"""

from typing import Any, Literal
import pandas as pd
import numpy as np

from src.analytics.advanced import detect_date_columns


def generate_time_series_forecast(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    forecast_periods: int = 6,
    frequency: Literal["D", "W", "M", "Y"] = "M",
    alpha: float = 0.3,
    beta: float = 0.1,
) -> dict[str, Any]:
    """Generate statistical baseline time-series forecasts with uncertainty intervals.

    Uses Holt's Linear Exponential Smoothing to extrapolate both level and trend components.

    Args:
        df: Input DataFrame.
        date_col: Timestamp/date column name.
        metric_col: Numerical metric column name.
        forecast_periods: Number of future periods to project (1 to 24).
        frequency: Aggregation frequency ('D'=Daily, 'W'=Weekly, 'M'=Monthly, 'Y'=Yearly).
        alpha: Level smoothing coefficient (0.05 to 0.95).
        beta: Trend smoothing coefficient (0.01 to 0.50).

    Returns:
        Structured dictionary containing historical observations, future forecasts,
        confidence bounds, and uncertainty explanations.
    """
    if df.empty or date_col not in df.columns or metric_col not in df.columns:
        return {
            "status": "error",
            "message": "Dataset is empty or required columns are missing.",
            "combined_df": pd.DataFrame(),
            "summary": {},
        }

    # Clean temporal subset
    sub = pd.DataFrame(
        {
            "raw_date": pd.to_datetime(df[date_col], errors="coerce"),
            "raw_metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()

    if sub.empty or len(sub) < 5:
        return {
            "status": "error",
            "message": f"Insufficient valid records ({len(sub)} found). Forecasting requires at least 5 valid observations.",
            "combined_df": pd.DataFrame(),
            "summary": {},
        }

    sub = sub.sort_values(by="raw_date")

    # Map frequency to period bucket
    if frequency == "D":
        sub["period_start"] = sub["raw_date"].dt.floor("D")
        sub["period"] = sub["raw_date"].dt.strftime("%Y-%m-%d")
        freq_offset = pd.Timedelta(days=1)
        freq_format = "%Y-%m-%d"
    elif frequency == "W":
        sub["period_start"] = sub["raw_date"].dt.to_period("W").apply(lambda r: r.start_time)
        sub["period"] = sub["period_start"].dt.strftime("%Y-%m-%d")
        freq_offset = pd.Timedelta(weeks=1)
        freq_format = "%Y-%m-%d"
    elif frequency == "Y":
        sub["period_start"] = sub["raw_date"].dt.to_period("Y").apply(lambda r: r.start_time)
        sub["period"] = sub["period_start"].dt.strftime("%Y")
        freq_offset = pd.DateOffset(years=1)
        freq_format = "%Y"
    else:  # default 'M' (Monthly)
        sub["period_start"] = sub["raw_date"].dt.to_period("M").apply(lambda r: r.start_time)
        sub["period"] = sub["period_start"].dt.strftime("%Y-%m")
        freq_offset = pd.DateOffset(months=1)
        freq_format = "%Y-%m"

    # Aggregate metric across periods
    agg_df = sub.groupby(["period", "period_start"])["raw_metric"].sum().reset_index()
    agg_df = agg_df.sort_values(by="period_start").reset_index(drop=True)
    agg_df = agg_df.rename(columns={"raw_metric": "actual"})

    n_hist = len(agg_df)
    if n_hist < 5:
        return {
            "status": "error",
            "message": f"Insufficient historical periods ({n_hist} periods found). A minimum of 5 historical periods is required for reliable statistical forecasting.",
            "combined_df": pd.DataFrame(),
            "summary": {},
        }

    y_vals = agg_df["actual"].values
    all_non_negative = (y_vals >= 0).all()

    # 1. Holt's Linear Exponential Smoothing fitting
    # Initialize level and trend
    level = y_vals[0]
    trend = y_vals[1] - y_vals[0]

    fitted_vals: list[float] = [level]
    residuals: list[float] = [0.0]

    for t in range(1, n_hist):
        prev_level = level
        prev_trend = trend
        y_t = y_vals[t]

        # Forecast of y_t made at t-1
        y_hat_t = prev_level + prev_trend
        fitted_vals.append(float(y_hat_t))
        residuals.append(float(y_t - y_hat_t))

        # Update Level and Trend
        level = alpha * y_t + (1.0 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * prev_trend

    agg_df["fitted"] = np.round(fitted_vals, 2)
    agg_df["forecast"] = np.nan
    agg_df["lower_bound"] = np.nan
    agg_df["upper_bound"] = np.nan
    agg_df["type"] = "Historical"

    # Residual Standard Deviation for prediction intervals
    res_arr = np.array(residuals[1:])  # exclude initial zero residual
    res_std = float(np.std(res_arr, ddof=1)) if len(res_arr) > 1 else 1.0

    # 2. Extrapolate future periods
    h_steps = int(np.clip(forecast_periods, 1, 24))
    future_rows: list[dict[str, Any]] = []
    last_period_date = agg_df.iloc[-1]["period_start"]

    for h in range(1, h_steps + 1):
        future_date = last_period_date + (freq_offset * h)
        future_period_str = future_date.strftime(freq_format)

        # Projected value
        y_pred_h = float(level + (h * trend))

        # Uncertainty expands with the forecast horizon: std * sqrt(h)
        uncertainty_margin = float(1.96 * res_std * np.sqrt(h))
        upper_bound = y_pred_h + uncertainty_margin
        lower_bound = y_pred_h - uncertainty_margin

        if all_non_negative:
            lower_bound = max(0.0, lower_bound)
            y_pred_h = max(0.0, y_pred_h)

        future_rows.append(
            {
                "period": future_period_str,
                "period_start": future_date,
                "actual": np.nan,
                "fitted": np.nan,
                "forecast": round(y_pred_h, 2),
                "lower_bound": round(upper_bound, 2) if lower_bound > upper_bound else round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "type": "Forecast",
            }
        )

    future_df = pd.DataFrame(future_rows)
    combined_df = pd.concat([agg_df, future_df], ignore_index=True)

    # Growth rate between last historical observation and final forecast
    last_actual = float(y_vals[-1])
    final_forecast = float(future_rows[-1]["forecast"])
    projected_change = final_forecast - last_actual
    projected_growth_pct = (projected_change / abs(last_actual) * 100.0) if last_actual != 0 else None

    direction = "Upward" if (projected_growth_pct and projected_growth_pct > 2.0) else (
        "Downward" if (projected_growth_pct and projected_growth_pct < -2.0) else "Stable"
    )

    explanation = (
        f"Generated a {h_steps}-period {frequency}-frequency projection using Holt's Linear Exponential Smoothing. "
        f"Historical trajectory ended at {last_actual:,.2f} and is projected to reach {final_forecast:,.2f} "
        + (f"({projected_growth_pct:+.1f}% shift, {direction} trajectory)" if projected_growth_pct is not None else "")
        + f". The 95% confidence intervals expand over the forecast horizon to account for compounding temporal uncertainty. "
        f"\n\n*Forecasting Notice & Limitations:* Projections extrapolate historical momentum and variance. "
        f"External macroeconomic shifts, seasonal outliers, or unexpected operational disruptions cannot be captured by purely univariate baselines."
    )

    summary = {
        "date_column": date_col,
        "metric_column": metric_col,
        "frequency": frequency,
        "historical_periods": n_hist,
        "forecast_periods": h_steps,
        "last_actual_value": round(last_actual, 2),
        "final_projected_value": round(final_forecast, 2),
        "projected_growth_pct": round(projected_growth_pct, 1) if projected_growth_pct is not None else None,
        "trend_direction": direction,
        "residual_std": round(res_std, 2),
        "model_name": "Holt's Linear Exponential Smoothing",
        "explanation": explanation,
    }

    return {
        "status": "ok",
        "message": "",
        "combined_df": combined_df,
        "historical_df": agg_df,
        "forecast_df": future_df,
        "summary": summary,
    }
