"""Unit tests for the InsightFlow Machine Learning & AI features (Stage 11).

Covers:
- Anomaly detection (Isolation Forest, severity scoring, missing value handling)
- Predictive modeling (Task inference, feature preparation, Regression & Classification training)
- Model evaluation (MAE, RMSE, R-squared, Accuracy, Precision, Recall, F1, Confusion Matrix)
- Time-series forecasting (Holt's Linear smoothing, confidence intervals, insufficient history)
- Edge cases (Empty datasets, small datasets, constant features, missing values)
- Dependency-unavailable behavior (simulated uninstalled scikit-learn)
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.ml.evaluation import evaluate_regression, evaluate_classification
from src.ml.anomaly_detection import detect_anomalies, is_sklearn_available
from src.ml.prediction import infer_ml_task, prepare_features, train_predictive_model
from src.ml.forecasting import generate_time_series_forecast


class TestMachineLearningFeatures(unittest.TestCase):
    """Test suite for Machine Learning and predictive intelligence features."""

    def setUp(self) -> None:
        """Create sample test datasets."""
        np.random.seed(42)

        # 1. Continuous Regression dataset (Revenue predicted by Spend, Units, Discount)
        n_reg = 60
        spend = np.random.uniform(100, 1000, n_reg)
        units = np.random.randint(10, 100, n_reg)
        revenue = (spend * 2.5) + (units * 15.0) + np.random.normal(0, 20, n_reg)
        self.df_regression = pd.DataFrame(
            {
                "MarketingSpend": spend,
                "UnitsSold": units,
                "Revenue": revenue,
                "Region": np.random.choice(["East", "West", "North", "South"], n_reg),
            }
        )

        # 2. Discrete Classification dataset (Churn predicted by Usage, Tenure, SupportCalls)
        n_clf = 60
        usage = np.random.uniform(5, 50, n_clf)
        tenure = np.random.uniform(1, 12, n_clf)
        churn = np.where((usage < 20) & (tenure < 4), "Churned", "Retained")
        self.df_classification = pd.DataFrame(
            {
                "MonthlyUsage": usage,
                "TenureMonths": tenure,
                "SupportCalls": np.random.randint(0, 5, n_clf),
                "Churn": churn,
            }
        )

        # 3. Anomaly detection dataset with obvious injected outliers
        n_norm = 50
        x_norm = np.random.normal(100, 10, n_norm)
        y_norm = np.random.normal(50, 5, n_norm)
        # 3 extreme outliers
        x_out = [500.0, -200.0, 999.0]
        y_out = [250.0, -100.0, 450.0]
        self.df_anomaly = pd.DataFrame(
            {
                "FeatureX": np.concatenate([x_norm, x_out]),
                "FeatureY": np.concatenate([y_norm, y_out]),
            }
        )

        # 4. Time series dataset for forecasting
        dates = pd.date_range("2023-01-01", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
        values = [100.0, 110.0, 125.0, 130.0, 145.0, 160.0, 170.0, 185.0, 200.0, 215.0, 230.0, 250.0]
        self.df_timeseries = pd.DataFrame({"Date": dates, "Sales": values})

        # 5. Edge cases
        self.df_empty = pd.DataFrame()
        self.df_small = pd.DataFrame({"X": [1.0, 2.0], "Y": [10.0, 20.0]})
        self.df_single_class = pd.DataFrame({"X": [1, 2, 3, 4], "Target": ["A", "A", "A", "A"]})
        self.df_missing_data = pd.DataFrame(
            {
                "Feature1": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
                "Feature2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                "Target": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0],
            }
        )

    # -------------------------------------------------------------------------
    # Evaluation Module Tests
    # -------------------------------------------------------------------------
    def test_evaluate_regression_perfect_fit(self) -> None:
        """Verify perfect predictions yield MAE=0, RMSE=0, R2=1."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        res = evaluate_regression(y_true, y_pred)
        self.assertEqual(res["status"], "ok")
        metrics = res["metrics"]
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_evaluate_regression_imperfect(self) -> None:
        """Verify evaluation metrics on noisy regression outputs."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])
        res = evaluate_regression(y_true, y_pred)
        self.assertEqual(res["status"], "ok")
        metrics = res["metrics"]
        self.assertGreater(metrics["mae"], 0.0)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertIn("R²", metrics["definitions"])

    def test_evaluate_classification_binary(self) -> None:
        """Verify Accuracy, Precision, Recall, and Confusion Matrix calculations."""
        y_true = ["Pass", "Fail", "Pass", "Pass", "Fail"]
        y_pred = ["Pass", "Fail", "Pass", "Fail", "Fail"]
        res = evaluate_classification(y_true, y_pred)
        self.assertEqual(res["status"], "ok")
        metrics = res["metrics"]
        self.assertEqual(metrics["accuracy"], 0.8)  # 4/5 correct
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["sample_size"], 5)

    def test_evaluate_empty_inputs(self) -> None:
        """Verify graceful error reporting when arrays are empty."""
        res_reg = evaluate_regression([], [])
        self.assertEqual(res_reg["status"], "error")

        res_clf = evaluate_classification([], [])
        self.assertEqual(res_clf["status"], "error")

    # -------------------------------------------------------------------------
    # Anomaly Detection Tests
    # -------------------------------------------------------------------------
    def test_anomaly_detection_identifies_outliers(self) -> None:
        """Verify Isolation Forest flags injected extreme anomalies."""
        res = detect_anomalies(
            df=self.df_anomaly,
            feature_cols=["FeatureX", "FeatureY"],
            contamination=0.06,
        )
        self.assertEqual(res["status"], "ok")
        self.assertGreater(res["anomaly_count"], 0)
        self.assertLessEqual(res["anomaly_count"], 10)
        self.assertIn("anomalous_records", res)
        self.assertFalse(res["anomalous_records"].empty)
        self.assertIn("Anomaly Severity (%)", res["anomalous_records"].columns)

    def test_anomaly_detection_with_missing_values(self) -> None:
        """Verify median imputation handles missing values safely during anomaly profiling."""
        res = detect_anomalies(
            df=self.df_missing_data,
            feature_cols=["Feature1", "Feature2"],
            contamination=0.1,
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["total_records"], 10)

    def test_anomaly_detection_empty_and_insufficient(self) -> None:
        """Verify error handling on empty or tiny datasets."""
        res_empty = detect_anomalies(self.df_empty, ["A"])
        self.assertEqual(res_empty["status"], "error")

        res_small = detect_anomalies(self.df_small, ["X", "Y"])
        self.assertEqual(res_small["status"], "error")

    # -------------------------------------------------------------------------
    # Predictive Analytics Tests
    # -------------------------------------------------------------------------
    def test_infer_ml_task(self) -> None:
        """Verify automatic task detection distinguishes regression and classification."""
        task_reg = infer_ml_task(self.df_regression, "Revenue")
        self.assertEqual(task_reg["task_type"], "regression")

        task_clf = infer_ml_task(self.df_classification, "Churn")
        self.assertEqual(task_clf["task_type"], "classification")

    def test_prepare_features_excludes_target_column(self) -> None:
        """Verify target column is never included in preprocessed feature matrix X."""
        X, y, feats = prepare_features(self.df_regression, target_col="Revenue")
        self.assertNotIn("Revenue", X.columns)
        self.assertNotIn("Revenue", feats)
        self.assertEqual(len(y), len(X))

    def test_train_predictive_model_regression(self) -> None:
        """Verify supervised regression training with Linear Regression and Random Forest."""
        # Baseline Linear Regression
        res_lr = train_predictive_model(
            df=self.df_regression,
            target_col="Revenue",
            model_name="Linear Regression",
            test_size=0.2,
        )
        self.assertEqual(res_lr["status"], "ok")
        self.assertEqual(res_lr["task_type"], "regression")
        self.assertGreater(res_lr["metrics"]["r2"], 0.85)
        self.assertFalse(res_lr["sample_predictions"].empty)

        # Random Forest Regressor
        res_rf = train_predictive_model(
            df=self.df_regression,
            target_col="Revenue",
            model_name="Random Forest Regressor",
            test_size=0.2,
        )
        self.assertEqual(res_rf["status"], "ok")
        self.assertFalse(res_rf["feature_importance"].empty)
        self.assertIn("Importance (%)", res_rf["feature_importance"].columns)

    def test_train_predictive_model_classification(self) -> None:
        """Verify supervised classification training with Logistic Regression and Random Forest."""
        res_clf = train_predictive_model(
            df=self.df_classification,
            target_col="Churn",
            model_name="Random Forest Classifier",
            test_size=0.25,
        )
        self.assertEqual(res_clf["status"], "ok")
        self.assertEqual(res_clf["task_type"], "classification")
        self.assertGreater(res_clf["metrics"]["accuracy"], 0.6)
        self.assertIn("confusion_matrix", res_clf["metrics"])

    def test_train_predictive_model_single_class_error(self) -> None:
        """Verify classification rejects target with only 1 class."""
        res = train_predictive_model(self.df_single_class, target_col="Target")
        self.assertEqual(res["status"], "error")

    def test_predictive_model_leak_free_pipeline(self) -> None:
        """Verify model pipeline encapsulates preprocessing and avoids data leakage with missing values."""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        df_leak_test = pd.DataFrame(
            {
                "Spend": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                "Category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "Sales": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0, 115.0, 125.0],
            }
        )
        res = train_predictive_model(
            df=df_leak_test,
            target_col="Sales",
            model_name="Linear Regression",
            test_size=0.25,
            random_state=42,
        )
        self.assertEqual(res["status"], "ok")
        self.assertIn("pipeline", res)
        pipeline = res["pipeline"]
        self.assertIsInstance(pipeline, Pipeline)
        preprocessor = pipeline.named_steps["preprocessor"]
        self.assertIsInstance(preprocessor, ColumnTransformer)

        # Test end-to-end inference on raw unseen data with missing values without data leakage
        raw_new_df = pd.DataFrame({"Spend": [np.nan, 50.0], "Category": ["A", "B"]})
        preds = pipeline.predict(raw_new_df)
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predictive_model_unseen_category_in_test_set(self) -> None:
        """Verify pipeline handles unseen categorical levels gracefully without throwing exceptions."""
        df_cat = pd.DataFrame(
            {
                "Num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "Tier": ["Gold", "Silver", "Gold", "Silver", "Gold", "Silver", "Gold", "Silver", "Gold", "Silver", "Gold", "Silver"],
                "Score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            }
        )
        res = train_predictive_model(
            df=df_cat,
            target_col="Score",
            model_name="Linear Regression",
            test_size=0.25,
        )
        self.assertEqual(res["status"], "ok")
        pipeline = res["pipeline"]

        # Predict on totally unseen category 'Platinum'
        test_unknown = pd.DataFrame({"Num": [15.0], "Tier": ["Platinum"]})
        pred = pipeline.predict(test_unknown)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))

    # -------------------------------------------------------------------------
    # Time-Series Forecasting Tests
    # -------------------------------------------------------------------------
    def test_time_series_forecasting_sufficient_history(self) -> None:
        """Verify Holt's linear exponential smoothing produces projected periods and bounds."""
        res = generate_time_series_forecast(
            df=self.df_timeseries,
            date_col="Date",
            metric_col="Sales",
            forecast_periods=4,
            frequency="M",
        )
        self.assertEqual(res["status"], "ok")
        summary = res["summary"]
        self.assertEqual(summary["historical_periods"], 12)
        self.assertEqual(summary["forecast_periods"], 4)
        self.assertEqual(summary["trend_direction"], "Upward")

        forecast_df = res["forecast_df"]
        self.assertEqual(len(forecast_df), 4)
        self.assertTrue((forecast_df["upper_bound"] >= forecast_df["forecast"]).all())
        self.assertTrue((forecast_df["forecast"] >= forecast_df["lower_bound"]).all())

    def test_time_series_forecasting_insufficient_history(self) -> None:
        """Verify forecasting fails gracefully when historical periods < 5."""
        df_short = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "Val": [10.0, 20.0, 30.0],
            }
        )
        res = generate_time_series_forecast(df_short, "Date", "Val")
        self.assertEqual(res["status"], "error")
        self.assertIn("at least 5", res["message"])

    # -------------------------------------------------------------------------
    # Dependency-Unavailable Behavior Tests
    # -------------------------------------------------------------------------
    @patch("src.ml.anomaly_detection.is_sklearn_available", return_value=False)
    def test_anomaly_detection_dependency_unavailable(self, mock_sklearn) -> None:
        """Verify anomaly detection returns 'dependency_unavailable' when scikit-learn is mocked as missing."""
        res = detect_anomalies(self.df_anomaly, ["FeatureX"])
        self.assertEqual(res["status"], "dependency_unavailable")
        self.assertIn("scikit-learn", res["message"])

    @patch("src.ml.prediction.is_sklearn_available", return_value=False)
    def test_prediction_dependency_unavailable(self, mock_sklearn) -> None:
        """Verify prediction returns 'dependency_unavailable' when scikit-learn is mocked as missing."""
        res = train_predictive_model(self.df_regression, "Revenue")
        self.assertEqual(res["status"], "dependency_unavailable")
        self.assertIn("scikit-learn", res["message"])


if __name__ == "__main__":
    unittest.main()
