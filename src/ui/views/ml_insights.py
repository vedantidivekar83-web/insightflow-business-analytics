"""Machine Learning & Predictive Intelligence dashboard view for InsightFlow.

Renders explainable, production-ready AI/ML workflows:
- Multivariate Anomaly Detection (Isolation Forest)
- Supervised Predictive Analytics (Auto Classification & Regression)
- Statistical Time-Series Forecasting (Holt's Linear Exponential Smoothing with Uncertainty Bands)
- Model Evaluation, Diagnostics, and Limitations Documentation
"""

from typing import Any
import streamlit as st
import pandas as pd
import numpy as np

from src.data.explorer import classify_columns
from src.analytics.advanced import detect_date_columns
from src.ml.anomaly_detection import is_sklearn_available, detect_anomalies
from src.ml.prediction import infer_ml_task, train_predictive_model
from src.ml.forecasting import generate_time_series_forecast
from src.visualization.ml_charts import (
    plot_anomaly_scatter,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_confusion_matrix_heatmap,
    plot_forecast_chart,
)


def render_ml_insights_view() -> None:
    """Render the Machine Learning & Predictive Analytics interface."""
    active_df: pd.DataFrame | None = st.session_state.get("dataset")
    metadata: dict[str, Any] | None = st.session_state.get("dataset_metadata")

    # 1. Empty State Guard
    if active_df is None or active_df.empty:
        st.markdown(
            """
            <div class="if-page-header">
                <h2 class="if-page-title">
                    Machine Learning & Predictive Intelligence
                </h2>
                <p class="if-page-subtitle">
                    Unsupervised anomaly detection, automated predictive modeling, statistical forecasting, and model explainability.
                </p>
            </div>
            <div class="if-placeholder-box">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🤖</div>
                <h3 style="margin: 0.4rem 0 0.35rem 0; font-size: 1.18rem; color: #F1F5F9; font-weight: 600;">
                    No Active Dataset for Machine Learning
                </h3>
                <p style="margin: 0 auto; max-width: 500px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                    Please navigate to <strong>Data Upload</strong> in the sidebar to load a dataset first.
                    Once active, InsightFlow will enable Isolation Forest anomaly detection, automated supervised
                    prediction (classification & regression), and forward-looking time-series forecasting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filename = metadata["filename"] if metadata else "Active Dataset"

    # Header section
    st.markdown(
        f"""
        <div class="if-page-header">
            <h2 class="if-page-title">
                Machine Learning & Predictive Intelligence
            </h2>
            <p class="if-page-subtitle">
                Active Dataset: <strong style="color: #22D3EE;">{filename}</strong>
                &bull; {len(active_df):,} records &bull; {len(active_df.columns)} dimensions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Classify dimensions
    classified = classify_columns(active_df)
    numerical_cols = classified["numerical"]
    categorical_cols = classified["categorical"]
    date_cols = detect_date_columns(active_df)

    # 2. Main Feature Tabs
    tab_anomaly, tab_pred, tab_forecast, tab_eval = st.tabs(
        [
            "🚨  Anomaly Detection",
            "🎯  Predictive Analytics",
            "📅  Time-Series Forecasting",
            "📊  Evaluation & Explainability",
        ]
    )

    # =========================================================================
    # TAB 1: Anomaly Detection (Isolation Forest)
    # =========================================================================
    with tab_anomaly:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Isolation Forest Anomaly Detection</h3>",
            unsafe_allow_html=True,
        )

        if not is_sklearn_available():
            st.warning(
                "⚠️ **Dependency Notice:** scikit-learn is required to execute Isolation Forest anomaly detection. "
                "Please ensure scikit-learn is installed in your Python environment."
            )
        elif len(numerical_cols) < 1:
            st.info("ℹ️ Anomaly detection requires at least one numerical feature.")
        else:
            anom_c1, anom_c2 = st.columns([3, 1])
            with anom_c1:
                sel_anom_features = st.multiselect(
                    "Select Numerical Features to Profile",
                    options=numerical_cols,
                    default=numerical_cols[: min(4, len(numerical_cols))],
                    key="anom_features_sel",
                    help="Isolation Forest isolates anomalies across multi-dimensional numerical coordinates.",
                )
            with anom_c2:
                sel_contam = st.slider(
                    "Expected Contamination Rate",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01,
                    format="%d%%",
                    key="anom_contam_sel",
                    help="Estimated proportion of outliers in the data (default 5%).",
                )

            if not sel_anom_features:
                st.info("Please select at least one numerical feature to run anomaly detection.")
            else:
                anom_res = detect_anomalies(
                    df=active_df,
                    feature_cols=sel_anom_features,
                    contamination=sel_contam,
                )

                if anom_res["status"] == "error":
                    st.warning(f"⚠️ {anom_res['message']}")
                elif anom_res["status"] == "ok":
                    am1, am2, am3, am4 = st.columns(4)
                    with am1:
                        st.metric(label="Evaluated Records", value=f"{anom_res['total_records']:,}")
                    with am2:
                        st.metric(label="Isolated Anomalies", value=f"{anom_res['anomaly_count']:,}")
                    with am3:
                        st.metric(label="Anomaly Rate", value=f"{anom_res['anomaly_pct']:.1f}%")
                    with am4:
                        st.metric(label="Typical Normal Records", value=f"{anom_res['normal_count']:,}")

                    # 2D Scatter if >= 2 features selected
                    if len(sel_anom_features) >= 2:
                        combined_anom_df = pd.concat([anom_res["normal_records"], anom_res["anomalous_records"]], axis=0)
                        fig_anom = plot_anomaly_scatter(
                            df=combined_anom_df,
                            x_col=sel_anom_features[0],
                            y_col=sel_anom_features[1],
                        )
                        st.plotly_chart(fig_anom, use_container_width=True)

                    # Explainability callout
                    st.info(f"💡 **Anomaly Diagnostics:** {anom_res['explanation']}")

                    # Anomalous records table
                    anom_table = anom_res["anomalous_records"]
                    if anom_table.empty:
                        st.success("✓ No records were flagged as severe anomalies under the selected contamination threshold.")
                    else:
                        st.markdown(
                            f"<h4 style='font-size: 1rem; font-weight: 600; color: #F8FAFC; margin: 1rem 0 0.5rem 0;'>Flagged Anomalous Records ({len(anom_table)} rows)</h4>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(anom_table, use_container_width=True)

    # =========================================================================
    # TAB 2: Predictive Analytics (Supervised Learning)
    # =========================================================================
    with tab_pred:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Supervised Predictive Modeling</h3>",
            unsafe_allow_html=True,
        )

        all_candidate_targets = [str(c) for c in active_df.columns if active_df[c].nunique() > 1]
        if not all_candidate_targets:
            st.info("ℹ️ Dataset has no suitable target variables with multiple distinct values.")
        else:
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                sel_target = st.selectbox("Select Target Variable to Predict", options=all_candidate_targets, key="pred_target_sel")
            with pc2:
                # Infer task
                task_diag = infer_ml_task(active_df, sel_target)
                inferred_task = task_diag.get("task_type", "regression")
                st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
                if inferred_task == "classification":
                    st.markdown(f"<span class='if-badge if-badge-mint'>🎯 Task: Classification ({task_diag.get('unique_count', 0)} classes)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='if-badge if-badge-cyan'>📈 Task: Regression (Continuous)</span>", unsafe_allow_html=True)
            with pc3:
                model_options = (
                    ["auto", "Random Forest Regressor", "Linear Regression"]
                    if inferred_task == "regression"
                    else ["auto", "Random Forest Classifier", "Logistic Regression"]
                )
                sel_model = st.selectbox("Predictive Algorithm", options=model_options, key="pred_model_sel")

            # Feature subset selection
            eligible_features = [c for c in active_df.columns if c != sel_target]
            sel_features = st.multiselect(
                "Predictor Features (Default: All Eligible)",
                options=eligible_features,
                default=eligible_features[: min(10, len(eligible_features))],
                key="pred_features_sel",
            )

            # Test split slider
            test_split = st.slider("Held-out Test Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05, format="%d%%", key="pred_test_split")

            if st.button("🚀 Train Model & Evaluate", use_container_width=True, key="pred_train_btn"):
                with st.spinner("Training predictive model and validating on held-out test data..."):
                    pred_res = train_predictive_model(
                        df=active_df,
                        target_col=sel_target,
                        feature_cols=sel_features,
                        model_name=sel_model,
                        test_size=test_split,
                    )

                if pred_res["status"] == "error":
                    st.warning(f"⚠️ {pred_res['message']}")
                elif pred_res["status"] == "ok":
                    st.session_state["_last_pred_result"] = pred_res

            # Display results if present
            pred_res = st.session_state.get("_last_pred_result")
            if pred_res and pred_res.get("target_col") == sel_target and pred_res["status"] == "ok":
                metrics = pred_res["metrics"]
                task = pred_res["task_type"]

                # Metric Cards
                if task == "regression":
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    with pm1:
                        st.metric(label="R² Goodness of Fit", value=f"{metrics['r2']:.3f}")
                    with pm2:
                        st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['mae']:,.2f}")
                    with pm3:
                        st.metric(label="Root Mean Squared Error", value=f"{metrics['rmse']:,.2f}")
                    with pm4:
                        st.metric(label="Test Sample Size", value=f"{metrics['sample_size']:,} records")
                else:
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    with pm1:
                        st.metric(label="Test Accuracy", value=f"{metrics['accuracy']*100:.1f}%")
                    with pm2:
                        st.metric(label="Macro Precision", value=f"{metrics['precision']*100:.1f}%")
                    with pm3:
                        st.metric(label="Macro Recall", value=f"{metrics['recall']*100:.1f}%")
                    with pm4:
                        st.metric(label="Macro F1-Score", value=f"{metrics['f1_score']*100:.1f}%")

                # Visualizations
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    imp_df = pred_res["feature_importance"]
                    if not imp_df.empty:
                        fig_imp = plot_feature_importance(imp_df, top_n=8)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("Feature importance extraction not supported for this model.")

                with v_col2:
                    if task == "regression":
                        sample_preds = pred_res["sample_predictions"]
                        fig_align = plot_actual_vs_predicted(sample_preds)
                        st.plotly_chart(fig_align, use_container_width=True)
                    else:
                        cm_df = metrics["confusion_matrix"]
                        fig_cm = plot_confusion_matrix_heatmap(cm_df)
                        st.plotly_chart(fig_cm, use_container_width=True)

                # Narrative explainability callout
                st.markdown(
                    f"""
                    <div class="if-callout">
                        <strong style="color: #22D3EE;">Model Diagnostic & Feature Dynamics:</strong><br>
                        {pred_res['explanation']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Sample test predictions table
                with st.expander("🔍 Inspect Sample Test Set Predictions vs Actual Values", expanded=False):
                    st.dataframe(pred_res["sample_predictions"], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: Time-Series Forecasting
    # =========================================================================
    with tab_forecast:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Statistical Time-Series Forecasting</h3>",
            unsafe_allow_html=True,
        )

        if not date_cols:
            st.info(
                "ℹ️ **No Date/Timestamp Column Detected:** Time-series forecasting requires a temporal coordinate column "
                "(e.g., Order Date, Timestamp, or Year)."
            )
        elif not numerical_cols:
            st.info("ℹ️ **No Numerical Metric Detected:** Forecasting requires a numerical metric to extrapolate into future horizons.")
        else:
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                sel_fc_date = st.selectbox("Date Dimension", options=date_cols, key="fc_date_sel")
            with fc2:
                sel_fc_num = st.selectbox("Metric to Forecast", options=numerical_cols, key="fc_num_sel")
            with fc3:
                fc_freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D", "Yearly": "Y"}
                sel_fc_freq_label = st.selectbox("Forecast Frequency", options=list(fc_freq_map.keys()), key="fc_freq_sel")
                actual_fc_freq = fc_freq_map[sel_fc_freq_label]
            with fc4:
                sel_fc_horizon = st.slider("Forecast Horizon (Steps)", min_value=1, max_value=18, value=6, key="fc_horizon_sel")

            fc_res = generate_time_series_forecast(
                df=active_df,
                date_col=sel_fc_date,
                metric_col=sel_fc_num,
                forecast_periods=sel_fc_horizon,
                frequency=actual_fc_freq,
            )

            if fc_res["status"] == "error":
                st.warning(f"⚠️ {fc_res['message']}")
            else:
                fc_sum = fc_res["summary"]
                growth_val = fc_sum.get("projected_growth_pct")
                growth_str = f"{growth_val:+.1f}%" if growth_val is not None else "N/A"

                # Metric Cards
                fm1, fm2, fm3, fm4 = st.columns(4)
                with fm1:
                    st.metric(label="Historical Base (Last Observed)", value=f"{fc_sum['last_actual_value']:,.2f}")
                with fm2:
                    st.metric(label=f"Projected Horizon ({sel_fc_horizon} {sel_fc_freq_label}s)", value=f"{fc_sum['final_projected_value']:,.2f}")
                with fm3:
                    st.metric(label="Projected Trajectory Shift", value=growth_str)
                with fm4:
                    st.metric(label="Residual Volatility (±σ)", value=f"±{fc_sum['residual_std']:,.2f}")

                # Forecast Chart with Confidence Band
                fig_fc = plot_forecast_chart(fc_res["combined_df"], metric_label=sel_fc_num)
                st.plotly_chart(fig_fc, use_container_width=True)

                # Narrative explainability
                st.info(f"💡 **Forecast Insights & Limitations:** {fc_sum['explanation']}")

                # Future projections table
                with st.expander(f"📋 View Projected {sel_fc_freq_label} Forecast Table ({len(fc_res['forecast_df'])} periods)", expanded=False):
                    disp_fc = fc_res["forecast_df"][["period", "forecast", "lower_bound", "upper_bound"]].copy()
                    disp_fc.columns = ["Forecast Period", f"Projected {sel_fc_num}", "95% Lower Bound", "95% Upper Bound"]
                    st.dataframe(disp_fc, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: Model Evaluation & Explainability
    # =========================================================================
    with tab_eval:
        st.markdown(
            "<h3 style='font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin: 0.5rem 0 0.85rem 0;'>Machine Learning Model Diagnostics & Evaluation Guide</h3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            This section provides transparency into how InsightFlow trains, evaluates, and validates machine learning models.
            All models run entirely locally on your CPU with strictly in-memory data processing.
            """
        )

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            with st.container(border=True):
                st.markdown("#### 📐 Regression Metrics & Interpretation")
                st.markdown(
                    """
                    - **Mean Absolute Error (MAE):** The average absolute difference between predicted and actual observed values. Expressed directly in the metric's original units.
                    - **Root Mean Squared Error (RMSE):** Calculates the square root of squared residuals, assigning disproportionate penalty to large errors.
                    - **Coefficient of Determination (R²):** The percentage of total target variance captured by feature variables. A score of 1.0 represents perfect prediction; 0.0 indicates baseline mean performance.
                    """
                )

            with st.container(border=True):
                st.markdown("#### 🌲 Algorithm Mechanics")
                st.markdown(
                    """
                    - **Isolation Forest:** Builds ensembles of random decision trees. Data points isolated near the root of the tree with fewer splits represent statistical anomalies.
                    - **Random Forest:** Trains an ensemble of diverse decision trees and averages their outputs (regression) or takes a majority vote (classification) to resist overfitting.
                    - **Linear / Logistic Regression:** Identifies baseline linear coefficients between predictors and the target variable.
                    """
                )

        with col_g2:
            with st.container(border=True):
                st.markdown("#### 🎯 Classification Metrics & Interpretation")
                st.markdown(
                    """
                    - **Accuracy:** The overall fraction of correctly classified observations ($Correct / Total$).
                    - **Precision:** The proportion of instances predicted as a class that were truly that class ($TP / (TP + FP)$). High precision minimizes false alarms.
                    - **Recall (Sensitivity):** The proportion of actual instances successfully captured by the model ($TP / (TP + FN)$). High recall minimizes missed cases.
                    - **F1-Score:** The harmonic mean of precision and recall, providing an honest assessment under class imbalances.
                    - **Confusion Matrix:** Cross-tabulates actual observed categories against predicted classifications.
                    """
                )

            with st.container(border=True):
                st.markdown("#### ⚠️ Operational Limitations & Best Practices")
                st.markdown(
                    """
                    - **Probabilistic Estimates:** Model outputs reflect historical mathematical correlations, not guarantees.
                    - **Concept Drift:** Changes in real-world business environments can cause model accuracy to decay over time.
                    - **Target Leakage Prevention:** InsightFlow automatically isolates and strips the target variable from training features to prevent leakage.
                    - **Human-in-the-Loop:** Predictions should guide domain experts rather than automate high-risk operational actions.
                    """
                )
