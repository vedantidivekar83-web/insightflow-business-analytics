"""Model evaluation metrics and plain-English explainability for InsightFlow.

Provides rigorous evaluation routines for both regression and classification tasks:
- Regression: MAE, RMSE, R-squared, residuals analysis
- Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

from typing import Any
import numpy as np
import pandas as pd


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate comprehensive regression evaluation metrics and explanations.

    Args:
        y_true: Ground truth observed values.
        y_pred: Model predicted values.

    Returns:
        Structured dictionary with MAE, RMSE, R-squared, residual stats, and explanations.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Filter out NaNs if any
    mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    y_t = y_true_arr[mask]
    y_p = y_pred_arr[mask]

    n = len(y_t)
    if n == 0:
        return {
            "status": "error",
            "message": "No valid numeric observations available to evaluate regression.",
            "metrics": {},
        }

    residuals = y_t - y_p
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # R-squared calculation: 1 - SS_res / SS_tot
    ss_res = float(np.sum(sq_errors))
    y_mean = float(np.mean(y_t))
    ss_tot = float(np.sum((y_t - y_mean) ** 2))

    if ss_tot > 0:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    mean_target = float(np.mean(y_t))
    mae_pct = (mae / abs(mean_target) * 100.0) if mean_target != 0 else None

    # Plain-English narrative explanation
    explanation = (
        f"The model explains {max(0.0, r2)*100:.1f}% of total variance in the target metric (R² = {r2:.3f}). "
        f"On average, individual predictions deviate from actual observed values by ±{mae:,.2f} units (MAE"
        + (f", representing ~{mae_pct:.1f}% of target average" if mae_pct is not None else "")
        + f"). The root mean squared error (RMSE) is {rmse:,.2f}, indicating the magnitude of larger errors."
    )

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "min_residual": round(float(np.min(residuals)), 4),
        "max_residual": round(float(np.max(residuals)), 4),
        "sample_size": n,
        "explanation": explanation,
        "definitions": {
            "MAE": "Mean Absolute Error: The average absolute difference between predicted and actual values in original metric units.",
            "RMSE": "Root Mean Squared Error: Gives higher penalty to large forecasting errors; ideal for spotting outlier deviations.",
            "R²": "Coefficient of Determination: The proportion of variance in the dependent variable explained by features (1.0 = perfect prediction).",
        },
    }

    return {
        "status": "ok",
        "message": "",
        "metrics": metrics,
    }


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate comprehensive classification evaluation metrics, confusion matrix, and explanations.

    Args:
        y_true: Ground truth class labels.
        y_pred: Model predicted class labels.

    Returns:
        Structured dictionary with Accuracy, Precision, Recall, F1, Confusion Matrix, and explanations.
    """
    y_t = pd.Series(y_true).astype(str).reset_index(drop=True)
    y_p = pd.Series(y_pred).astype(str).reset_index(drop=True)

    mask = (y_t.notna()) & (y_p.notna()) & (y_t != "nan") & (y_p != "nan")
    y_t = y_t[mask]
    y_p = y_p[mask]

    n = len(y_t)
    if n == 0:
        return {
            "status": "error",
            "message": "No valid observations available to evaluate classification.",
            "metrics": {},
        }

    classes = sorted(list(set(y_t.unique()) | set(y_p.unique())))

    # Accuracy
    correct = (y_t == y_p).sum()
    accuracy = float(correct / n)

    # Class-level metrics
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    class_breakdown: list[dict[str, Any]] = []

    for c in classes:
        tp = float(((y_p == c) & (y_t == c)).sum())
        fp = float(((y_p == c) & (y_t != c)).sum())
        fn = float(((y_p != c) & (y_t == c)).sum())
        support = float((y_t == c).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        class_breakdown.append(
            {
                "Class": str(c),
                "Support": int(support),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
            }
        )

    # Macro averages
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Confusion matrix DataFrame
    cm_df = pd.crosstab(
        y_t,
        y_p,
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    # Ensure all classes are represented in both axes
    for c in classes:
        if c not in cm_df.index:
            cm_df.loc[c] = 0
        if c not in cm_df.columns:
            cm_df[c] = 0
    cm_df = cm_df.loc[classes, classes].fillna(0).astype(int)

    explanation = (
        f"The model achieved an overall accuracy of {accuracy*100:.1f}%, correctly predicting "
        f"{correct:,} of {n:,} test records. "
        f"Macro Precision is {macro_precision*100:.1f}% (reliable positive identifications) and "
        f"Macro Recall is {macro_recall*100:.1f}% (ability to find all true instances), yielding a "
        f"balanced Macro F1-Score of {macro_f1*100:.1f}%."
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1_score": round(macro_f1, 4),
        "classes": classes,
        "sample_size": n,
        "class_breakdown": pd.DataFrame(class_breakdown),
        "confusion_matrix": cm_df,
        "explanation": explanation,
        "definitions": {
            "Accuracy": "The proportion of total predictions that the model classified correctly.",
            "Precision": "Out of all instances predicted as a given class, how many were truly that class.",
            "Recall": "Out of all actual instances of a given class, how many the model successfully captured.",
            "F1-Score": "Harmonic mean of Precision and Recall; essential for evaluating imbalanced classes.",
        },
    }

    return {
        "status": "ok",
        "message": "",
        "metrics": metrics,
    }
