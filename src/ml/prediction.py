"""Supervised Predictive Analytics engine for InsightFlow.

Provides automated task inference (Classification vs Regression),
data preprocessing, train/test splitting, baseline and ensemble model training,
feature importance extraction, and explainability summaries.
"""

from typing import Any, Literal
import pandas as pd
import numpy as np

from src.ml.evaluation import evaluate_regression, evaluate_classification


def is_sklearn_available() -> bool:
    """Check whether scikit-learn is installed and importable."""
    try:
        import sklearn.linear_model  # noqa: F401
        import sklearn.ensemble  # noqa: F401
        return True
    except ImportError:
        return False


def infer_ml_task(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Inspect target column and determine whether task is Classification or Regression.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        Dictionary with 'task_type' ('classification' | 'regression'), unique counts, and rationale.
    """
    if df.empty or target_col not in df.columns:
        return {
            "status": "error",
            "task_type": "unknown",
            "message": f"Target column '{target_col}' not found or dataset is empty.",
        }

    series = df[target_col].dropna()
    if series.empty:
        return {
            "status": "error",
            "task_type": "unknown",
            "message": f"Target column '{target_col}' contains only missing values.",
        }

    nunique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # 1. Non-numeric or boolean is always classification
    if not is_numeric or pd.api.types.is_bool_dtype(series):
        classes = sorted(list(series.astype(str).unique()))
        return {
            "status": "ok",
            "task_type": "classification",
            "target_col": target_col,
            "unique_count": nunique,
            "classes": classes,
            "rationale": f"Target '{target_col}' is categorical/discrete with {nunique} unique classes: {', '.join(classes[:5])}.",
        }

    # 2. Numeric target: check if continuous or discrete class-like (<= 10 integers)
    is_all_integers = (series % 1 == 0).all()
    if nunique <= 10 and is_all_integers:
        classes = sorted([str(int(x)) for x in series.unique()])
        return {
            "status": "ok",
            "task_type": "classification",
            "target_col": target_col,
            "unique_count": nunique,
            "classes": classes,
            "rationale": f"Target '{target_col}' contains discrete integer codes with {nunique} levels ({', '.join(classes[:5])}).",
        }

    return {
        "status": "ok",
        "task_type": "regression",
        "target_col": target_col,
        "unique_count": nunique,
        "classes": [],
        "rationale": f"Target '{target_col}' contains continuous numerical values spanning from {series.min():,.2f} to {series.max():,.2f}.",
    }


def build_preprocessing_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> Any:
    """Build a Scikit-learn ColumnTransformer for leak-free preprocessing.

    Applies median imputation to numerical features and most-frequent imputation
    plus one-hot encoding (with drop='first' and handle_unknown='ignore') to
    categorical features.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    transformers = []
    if numerical_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Clean, impute, and encode feature columns for standalone exploratory analysis.

    Note:
        This is a standalone exploratory feature-extraction utility. For supervised
        model training and held-out evaluation, use train_predictive_model() which
        encapsulates preprocessing inside a leak-free Scikit-learn Pipeline.

    Args:
        df: Input DataFrame.
        target_col: Name of the target variable (strictly excluded from features).
        feature_cols: Optional subset of feature column names.

    Returns:
        tuple of (preprocessed_feature_matrix_X, clean_target_series_y, feature_names).
    """
    if df.empty or target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=object), []

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        return pd.DataFrame(), pd.Series(dtype=object), []

    # Determine candidate features (excluding target to prevent leakage)
    if feature_cols:
        candidate_cols = [c for c in feature_cols if c in clean_df.columns and c != target_col]
    else:
        # Auto-select features: drop target, high-cardinality ID columns, and constant columns
        candidate_cols = []
        for col in clean_df.columns:
            if col == target_col:
                continue
            col_s = clean_df[col]
            # Exclude constant columns
            if col_s.nunique() <= 1:
                continue
            # Exclude high-cardinality string IDs (e.g. 100% unique strings like UUIDs)
            if col_s.dtype == object and col_s.nunique() == len(clean_df) and len(clean_df) > 15:
                continue
            candidate_cols.append(col)

    if not candidate_cols:
        return pd.DataFrame(), pd.Series(dtype=object), []

    X_raw = clean_df[candidate_cols].copy()
    y_raw = clean_df[target_col].copy()

    if not is_sklearn_available():
        return pd.DataFrame(), y_raw, []

    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(X_raw[c])]
    cat_cols = [c for c in candidate_cols if c not in num_cols]

    for c in cat_cols:
        X_raw[c] = X_raw[c].astype(object)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    try:
        X_trans = preprocessor.fit_transform(X_raw)
        feature_names = [f.split("__", 1)[-1] for f in preprocessor.get_feature_names_out()]
        X_encoded = pd.DataFrame(X_trans, columns=feature_names, index=X_raw.index)
        return X_encoded, y_raw, feature_names
    except Exception:
        return pd.DataFrame(), y_raw, []


def train_predictive_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    model_name: Literal["auto", "Linear Regression", "Random Forest Regressor", "Logistic Regression", "Random Forest Classifier"] = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a supervised ML model, evaluate on held-out test data, and extract feature importances.

    Imputation and categorical encoding are encapsulated inside a Scikit-learn Pipeline
    and fit exclusively on training data to prevent data leakage.

    Args:
        df: Input DataFrame.
        target_col: Column to predict.
        feature_cols: Optional list of predictor features.
        model_name: Model algorithm name or 'auto'.
        test_size: Proportion of dataset reserved for testing (0.1 to 0.4).
        random_state: Reproducibility seed.

    Returns:
        Structured dictionary containing model evaluation metrics, feature importances,
        test predictions comparison, and explainability notes.
    """
    if not is_sklearn_available():
        return {
            "status": "dependency_unavailable",
            "message": "The 'scikit-learn' library is required to train predictive models.",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    task_info = infer_ml_task(df, target_col)
    if task_info["status"] != "ok":
        return {
            "status": "error",
            "message": task_info.get("message", "Failed to inspect target variable."),
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    task_type = task_info["task_type"]

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        return {
            "status": "error",
            "message": "Dataset contains no valid records after dropping missing target values.",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    # Determine candidate features (excluding target to prevent leakage)
    if feature_cols:
        candidate_cols = [c for c in feature_cols if c in clean_df.columns and c != target_col]
    else:
        # Auto-select features: drop target, high-cardinality ID columns, and constant columns
        candidate_cols = []
        for col in clean_df.columns:
            if col == target_col:
                continue
            col_s = clean_df[col]
            # Exclude constant columns
            if col_s.nunique() <= 1:
                continue
            # Exclude high-cardinality string IDs (e.g. 100% unique strings like UUIDs)
            if col_s.dtype == object and col_s.nunique() == len(clean_df) and len(clean_df) > 15:
                continue
            candidate_cols.append(col)

    if not candidate_cols or len(clean_df) < 10:
        return {
            "status": "error",
            "message": f"Insufficient data ({len(clean_df)} rows, {len(candidate_cols)} candidate features). Predictive modeling requires at least 10 observations and 1 predictor feature.",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    X_raw = clean_df[candidate_cols].copy()
    y_raw = clean_df[target_col].copy()

    # Verify classification has at least 2 distinct classes in target
    if task_type == "classification" and y_raw.nunique() < 2:
        return {
            "status": "error",
            "message": f"Target '{target_col}' has only {y_raw.nunique()} unique class. Classification requires at least 2 distinct classes.",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.pipeline import Pipeline

    # Train / Test Split BEFORE preprocessing to strictly prevent data leakage
    test_frac = float(np.clip(test_size, 0.1, 0.4))
    stratify = y_raw if (task_type == "classification" and (y_raw.value_counts() >= 2).all()) else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_frac,
            random_state=random_state,
            stratify=stratify,
        )
    except Exception:
        # Fallback without stratification if rare classes exist
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_frac,
            random_state=random_state,
        )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Separate numerical and categorical columns
    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in candidate_cols if c not in num_cols]

    for c in cat_cols:
        X_train[c] = X_train[c].astype(object)
        X_test[c] = X_test[c].astype(object)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    # Model Selection & Instantiation
    actual_model_name = model_name
    if task_type == "regression":
        if model_name in ["auto", "Random Forest Regressor"]:
            estimator = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state)
            actual_model_name = "Random Forest Regressor"
        else:
            estimator = LinearRegression()
            actual_model_name = "Linear Regression"
    else:  # classification
        # Convert target to string class labels
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
        if model_name in ["auto", "Random Forest Classifier"]:
            estimator = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
            actual_model_name = "Random Forest Classifier"
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            actual_model_name = "Logistic Regression"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    # Fit Pipeline exclusively on training data and evaluate on held-out test data
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model training failed: {str(e)}",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    # Extract transformed feature names
    trained_preprocessor = pipeline.named_steps["preprocessor"]
    trained_model = pipeline.named_steps["model"]
    feature_names = [f.split("__", 1)[-1] for f in trained_preprocessor.get_feature_names_out()]

    if len(feature_names) == 0:
        return {
            "status": "error",
            "message": "No valid predictor features remained after preprocessing.",
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    # Evaluate Model
    if task_type == "regression":
        eval_result = evaluate_regression(y_test, y_pred)
    else:
        eval_result = evaluate_classification(y_test, y_pred)

    if eval_result["status"] != "ok":
        return {
            "status": "error",
            "message": eval_result.get("message", "Model evaluation failed."),
            "metrics": {},
            "feature_importance": pd.DataFrame(),
            "sample_predictions": pd.DataFrame(),
            "explanation": "",
        }

    metrics = eval_result["metrics"]

    # Extract Feature Importances / Coefficients
    importances_list: list[dict[str, Any]] = []
    if hasattr(trained_model, "feature_importances_"):
        raw_imp = trained_model.feature_importances_
        total_imp = raw_imp.sum() if raw_imp.sum() > 0 else 1.0
        for feat, val in zip(feature_names, raw_imp):
            importances_list.append(
                {
                    "Feature": feat,
                    "Importance (%)": round(float((val / total_imp) * 100.0), 2),
                }
            )
    elif hasattr(trained_model, "coef_"):
        coef_arr = np.abs(trained_model.coef_)
        if coef_arr.ndim > 1:
            coef_arr = coef_arr.mean(axis=0)
        total_coef = coef_arr.sum() if coef_arr.sum() > 0 else 1.0
        for feat, val in zip(feature_names, coef_arr):
            importances_list.append(
                {
                    "Feature": feat,
                    "Importance (%)": round(float((val / total_coef) * 100.0), 2),
                }
            )

    imp_df = pd.DataFrame(importances_list)
    if not imp_df.empty:
        imp_df = imp_df.sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)

    # Sample Predictions Comparison Table (First 15 test cases)
    sample_size = min(15, len(y_test))
    if task_type == "regression":
        sample_df = pd.DataFrame(
            {
                "Actual Value": np.round(np.asarray(y_test.iloc[:sample_size], dtype=float), 2),
                "Model Prediction": np.round(np.asarray(y_pred[:sample_size], dtype=float), 2),
            }
        )
        sample_df["Error (Abs)"] = np.round(np.abs(sample_df["Actual Value"] - sample_df["Model Prediction"]), 2)
    else:
        sample_df = pd.DataFrame(
            {
                "Actual Class": np.asarray(y_test.iloc[:sample_size], dtype=str),
                "Model Prediction": np.asarray(y_pred[:sample_size], dtype=str),
            }
        )
        sample_df["Correct?"] = np.where(sample_df["Actual Class"] == sample_df["Model Prediction"], "✓ Yes", "✗ No")

    # Explainability & Limitations Summary
    top_feature_str = (
        f"Top driving features are: {', '.join(imp_df.head(3)['Feature'].tolist())}."
        if not imp_df.empty
        else ""
    )

    narrative = (
        f"Trained a **{actual_model_name}** ({task_type.capitalize()} task) to predict target **'{target_col}'** "
        f"using {len(feature_names)} features across {len(X_train):,} training records and {len(X_test):,} held-out test records. "
        f"{metrics['explanation']} {top_feature_str} "
        f"\n\n*Important Model Limitations:* Predictions are probabilistic mathematical estimates based on observed historical features. "
        f"Model accuracy can deteriorate if underlying business conditions shift or if new unrecorded variables influence outcomes. "
        f"Predictions should assist human decisions, not replace operational domain judgment."
    )

    return {
        "status": "ok",
        "message": "",
        "task_type": task_type,
        "model_name": actual_model_name,
        "target_col": target_col,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(feature_names),
        "features_used": feature_names,
        "metrics": metrics,
        "feature_importance": imp_df,
        "sample_predictions": sample_df,
        "explanation": narrative,
        "pipeline": pipeline,
    }
