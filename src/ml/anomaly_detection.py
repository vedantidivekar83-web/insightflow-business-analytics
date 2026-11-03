"""Isolation Forest anomaly detection engine for InsightFlow.

Identifies multivariate statistical outliers across numerical dimensions
using scikit-learn's Isolation Forest algorithm with robust fallback handling.
"""

from typing import Any
import pandas as pd
import numpy as np


def is_sklearn_available() -> bool:
    """Check whether scikit-learn is installed and importable."""
    try:
        import sklearn.ensemble  # noqa: F401
        return True
    except ImportError:
        return False


def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Detect anomalous multi-dimensional records using Isolation Forest.

    Args:
        df: Input DataFrame (will not be mutated).
        feature_cols: List of numerical columns to analyze.
        contamination: Expected proportion of outliers (0.01 to 0.50).
        random_state: Random seed for deterministic tree generation.

    Returns:
        Structured dictionary containing anomaly counts, percentages,
        flagged records, and plain-English explainability notes.
    """
    if not is_sklearn_available():
        return {
            "status": "dependency_unavailable",
            "message": (
                "The 'scikit-learn' library is required to perform Isolation Forest anomaly detection. "
                "Please ensure scikit-learn is installed in your Python environment."
            ),
            "total_records": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_pct": 0.0,
            "anomalous_records": pd.DataFrame(),
            "features_used": [],
            "explanation": "",
        }

    if df.empty or not feature_cols:
        return {
            "status": "error",
            "message": "Dataset is empty or no numerical feature columns were specified.",
            "total_records": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_pct": 0.0,
            "anomalous_records": pd.DataFrame(),
            "features_used": [],
            "explanation": "",
        }

    # Verify features exist and contain numeric data
    valid_features = [c for c in feature_cols if c in df.columns]
    if not valid_features:
        return {
            "status": "error",
            "message": "None of the specified feature columns exist in the active dataset.",
            "total_records": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_pct": 0.0,
            "anomalous_records": pd.DataFrame(),
            "features_used": [],
            "explanation": "",
        }

    # Extract numerical matrix and impute missing values with column median
    sub_df = df[valid_features].copy()
    for col in valid_features:
        sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")

    # Drop rows that are completely NaN across all selected features
    valid_mask = sub_df.notna().any(axis=1)
    clean_sub = sub_df[valid_mask].copy()

    if len(clean_sub) < 4:
        return {
            "status": "error",
            "message": f"Insufficient observations ({len(clean_sub)} records). Anomaly detection requires at least 4 valid rows.",
            "total_records": len(clean_sub),
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_pct": 0.0,
            "anomalous_records": pd.DataFrame(),
            "features_used": valid_features,
            "explanation": "",
        }

    # Median imputation for isolated missing cells
    clean_sub = clean_sub.fillna(clean_sub.median())

    # Bound contamination rate safely
    clamped_contamination = float(np.clip(contamination, 0.01, 0.50))

    try:
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(
            contamination=clamped_contamination,
            random_state=random_state,
            n_estimators=100,
        )
        preds = iso_forest.fit_predict(clean_sub)
        scores = iso_forest.decision_function(clean_sub)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Isolation Forest failed during training: {str(e)}",
            "total_records": len(clean_sub),
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_pct": 0.0,
            "anomalous_records": pd.DataFrame(),
            "features_used": valid_features,
            "explanation": "",
        }

    # Flag: -1 indicates anomaly in IsolationForest, +1 indicates normal
    is_anomaly = (preds == -1)
    anomaly_count = int(is_anomaly.sum())
    total_records = len(clean_sub)
    normal_count = total_records - anomaly_count
    anomaly_pct = (anomaly_count / total_records) * 100.0 if total_records > 0 else 0.0

    # Calculate Anomaly Severity Score (0% = typical, 100% = most isolated outlier)
    # Decision function returns lower values for anomalies (often negative)
    min_score, max_score = scores.min(), scores.max()
    score_range = max_score - min_score if (max_score - min_score) > 0 else 1.0
    # Invert so higher score = higher severity
    normalized_severity = ((max_score - scores) / score_range) * 100.0

    # Build output table preserving original row indices
    result_df = df.loc[clean_sub.index].copy()
    result_df["_is_anomaly"] = is_anomaly
    result_df["Anomaly Severity (%)"] = np.round(normalized_severity, 1)

    anomalous_df = result_df[result_df["_is_anomaly"]].copy()
    anomalous_df = anomalous_df.sort_values(by="Anomaly Severity (%)", ascending=False)
    anomalous_df = anomalous_df.drop(columns=["_is_anomaly"])

    normal_df = result_df[~result_df["_is_anomaly"]].copy().drop(columns=["_is_anomaly"])

    explanation = (
        f"Isolation Forest evaluated {total_records:,} records across {len(valid_features)} numerical features "
        f"({', '.join(valid_features)}) and isolated {anomaly_count:,} anomalies ({anomaly_pct:.1f}% of observations). "
        f"The algorithm isolates outliers by constructing random decision trees; records requiring fewer tree splits "
        f"are flagged as statistically anomalous. "
        f"Important Notice: Anomalous records represent unusual multivariate combinations or extreme values; "
        f"they indicate high-interest operational events or data entry outliers, not necessarily fraud or errors."
    )

    return {
        "status": "ok",
        "message": "",
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_pct": round(anomaly_pct, 2),
        "contamination": clamped_contamination,
        "features_used": valid_features,
        "anomalous_records": anomalous_df,
        "normal_records": normal_df,
        "explanation": explanation,
    }
