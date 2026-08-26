
# Streamlit App: Bank Marketing Classification App
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. kNN
# 4. Naive Bayes
# 5. Random Forest


import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)



# Page Configuration


st.set_page_config(
    page_title="Bank Marketing Classification Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)



# CSS for UI


st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        .hero-section {
            padding: 28px 32px;
            border-radius: 18px;
            background: linear-gradient(135deg, #102a43 0%, #1d4ed8 55%, #14b8a6 100%);
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        }

        .hero-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            font-size: 17px;
            opacity: 0.95;
            line-height: 1.55;
            margin-bottom: 0px;
        }

        .section-card {
            background-color: white;
            padding: 22px 24px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }

        .small-card {
            background-color: white;
            padding: 18px 20px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            min-height: 104px;
        }

        .small-label {
            color: #64748b;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .small-value {
            color: #0f172a;
            font-size: 26px;
            font-weight: 800;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            padding: 18px 20px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .metric-label-custom {
            color: #64748b;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .metric-value-custom {
            color: #0f172a;
            font-size: 30px;
            font-weight: 850;
            letter-spacing: -0.3px;
        }

        .good-note {
            background-color: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 14px 16px;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 16px;
        }

        .footer-note {
            text-align: center;
            color: #64748b;
            font-size: 13px;
            padding-top: 18px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 26px;
        }

        div[data-testid="stFileUploader"] section {
            border-radius: 14px;
        }


    </style>
    """,
    unsafe_allow_html=True,
)



# Utility Functions


def convert_labels_to_binary(labels):
    """
    Converts labels into binary 0/1 format.
    Handles yes/no, 0/1, strings, pandas StringDtype, and numeric labels safely.
    """
    label_series = pd.Series(labels).copy()
    normalized = label_series.astype(str).str.strip().str.lower()

    mapping = {
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
    }

    return normalized.map(mapping)


@st.cache_resource
def load_model(model_path):
    """Loads a saved machine learning model."""
    return joblib.load(model_path)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label-custom">{label}</div>
            <div class="metric-value-custom">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_card(label, value):
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-label">{label}</div>
            <div class="small-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Model File Mapping


MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear and interpretable baseline classifier.",
    "Decision Tree": "Tree-based model that captures non-linear decision rules.",
    "kNN": "Instance-based model using nearest neighbor similarity.",
    "Naive Bayes": "Fast probabilistic classifier based on Bayes theorem.",
    "Random Forest": "Ensemble model using multiple decision trees for stable predictions.",
}



# Header / Hero Section

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">📊 Bank Marketing Classification Dashboard</div>
        <div class="hero-subtitle">
            Upload a CSV file, select a model, and evaluate prediction results using Accuracy, AUC, Precision, Recall, F1 Score, MCC, Confusion Matrix, and Classification Report.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



# Sidebar Controls


st.sidebar.title("⚙️ App Controls")
st.sidebar.caption("Use this panel to upload a CSV file and select the model for prediction.")

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_PATHS.keys()),
)

st.sidebar.markdown(
    f"""
    **Selected Model:**  
    `{selected_model_name}`

    {MODEL_DESCRIPTIONS[selected_model_name]}
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File to Predict Results",
    type=["csv"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    Upload a CSV file containing:

    - All input feature columns
    - Target column named `y`
    - Target values as `yes/no` or `0/1`
    """
)



# Main App Logic

if uploaded_file is not None:
    try:
       
        # Read uploaded CSV
       
        data = pd.read_csv(uploaded_file)

        if "y" not in data.columns:
            st.error("The uploaded CSV must contain a target column named 'y'.")
            st.stop()

        X_test = data.drop("y", axis=1)
        y_test_original = data["y"]
        y_test = convert_labels_to_binary(y_test_original)

        if y_test.isnull().sum() > 0:
            st.error("Target column 'y' should contain only yes/no or 0/1 values.")
            st.write("Unique values found in y column:", list(pd.Series(y_test_original).unique()))
            st.stop()

        y_test = y_test.astype(int).values

  
        # Load selected model
      
        model_path = MODEL_PATHS[selected_model_name]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = load_model(model_path)

        st.markdown(
            f"""
            <div class="good-note">
                ✅ {selected_model_name} model loaded successfully. Evaluation completed on uploaded test dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        
        # Predict labels and probabilities
        
        y_pred_raw = model.predict(X_test)
        y_pred = convert_labels_to_binary(y_pred_raw)

        if y_pred.isnull().sum() > 0:
            st.error("Prediction labels could not be converted properly.")
            st.write("Unique predicted values found:", list(pd.Series(y_pred_raw).unique()))
            st.stop()

        y_pred = y_pred.astype(int).values

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        
        # Calculate Metrics
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        
        # Dataset Summary Cards
        
        st.subheader("📌 Dataset and Model Summary")
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            summary_card("Records", f"{data.shape[0]:,}")
        with s2:
            summary_card("Input Features", f"{X_test.shape[1]}")
        with s3:
            summary_card("Selected Model", selected_model_name)
        with s4:
            positive_rate = pd.Series(y_test).mean() * 100
            summary_card("Positive Class", f"{positive_rate:.1f}%")

        
        # Uploaded Data Preview
       
        with st.expander("📁 View Uploaded Test Data Preview", expanded=False):
            st.dataframe(data.head(20), use_container_width=True)

        
        # Tabs for Results
        
        tab_metrics, tab_cm, tab_report, tab_predictions = st.tabs(
            [
                "📈 Selected Model Metrics",
                "🔷 Confusion Matrix",
                "📋 Classification Report",
                "🔍 Predictions",
            ]
        )

        with tab_metrics:
            st.subheader("📈 Evaluation Metrics")
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)

            with c1:
                metric_card("Accuracy", f"{accuracy:.4f}")
            with c2:
                metric_card("AUC Score", f"{auc:.4f}")
            with c3:
                metric_card("Precision", f"{precision:.4f}")
            with c4:
                metric_card("Recall", f"{recall:.4f}")
            with c5:
                metric_card("F1 Score", f"{f1:.4f}")
            with c6:
                metric_card("MCC Score", f"{mcc:.4f}")

            st.markdown(
                """
                **Interpretation Tip:** For this dataset, the target class is usually imbalanced. Therefore, AUC, Recall, F1 Score, and MCC should be reviewed along with Accuracy.
                """
            )

        with tab_cm:
            st.subheader("🔷 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted No", "Predicted Yes"],
                yticklabels=["Actual No", "Actual Yes"],
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title(f"Confusion Matrix - {selected_model_name}")
            st.pyplot(fig)

        with tab_report:
            st.subheader("📋 Classification Report")
            report = classification_report(
                y_test,
                y_pred,
                target_names=["no", "yes"],
                zero_division=0,
                output_dict=True,
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)

        with tab_predictions:
            st.subheader("🔍 Prediction Results")
            output_df = data.copy()
            output_df["Actual"] = pd.Series(y_test).map({0: "no", 1: "yes"}).values
            output_df["Predicted"] = pd.Series(y_pred).map({0: "no", 1: "yes"}).values

            st.dataframe(output_df.head(50), use_container_width=True)

            csv_output = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Prediction Results as CSV",
                data=csv_output,
                file_name=f"{selected_model_name.replace(' ', '_').lower()}_prediction_results.csv",
                mime="text/csv",
            )

        st.markdown("---")
        with st.expander("🏆 Advanced Analysis: Compare All Models", expanded=False):
            st.subheader("📊 Comparison of All Models")
            st.markdown(
                """
                This is an optional benchmarking section. Click the button below to evaluate all saved models on the uploaded CSV file and compare their performance side by side.
                """
            )

            if st.button("🚀 Run All Model Comparison", type="primary", use_container_width=True):
                comparison_results = []

                with st.spinner("Evaluating all models on uploaded CSV file..."):
                    for comparison_model_name, comparison_model_path in MODEL_PATHS.items():
                        if not os.path.exists(comparison_model_path):
                            st.warning(f"Model file not found: {comparison_model_path}")
                            continue

                        comparison_model = load_model(comparison_model_path)

                        comparison_pred_raw = comparison_model.predict(X_test)
                        comparison_pred = convert_labels_to_binary(comparison_pred_raw)

                        if comparison_pred.isnull().sum() > 0:
                            st.warning(f"Prediction labels could not be converted for {comparison_model_name}.")
                            continue

                        comparison_pred = comparison_pred.astype(int).values

                        if hasattr(comparison_model, "predict_proba"):
                            comparison_prob = comparison_model.predict_proba(X_test)[:, 1]
                        else:
                            comparison_prob = comparison_pred

                        comparison_results.append(
                            {
                                "ML Model Name": comparison_model_name,
                                "Accuracy": accuracy_score(y_test, comparison_pred),
                                "AUC": roc_auc_score(y_test, comparison_prob),
                                "Precision": precision_score(y_test, comparison_pred, zero_division=0),
                                "Recall": recall_score(y_test, comparison_pred, zero_division=0),
                                "F1": f1_score(y_test, comparison_pred, zero_division=0),
                                "MCC": matthews_corrcoef(y_test, comparison_pred),
                            }
                        )

                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results)
                    comparison_df = comparison_df.sort_values(by="AUC", ascending=False)
                    rounded_comparison_df = comparison_df.copy()

                    for metric_col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
                        rounded_comparison_df[metric_col] = rounded_comparison_df[metric_col].round(4)

                    st.markdown("### Model Comparison Table")
                    st.dataframe(rounded_comparison_df, use_container_width=True)

                    winner = rounded_comparison_df.iloc[0]
                    st.success(
                        f"🏆 Best model by AUC on uploaded CSV file: {winner['ML Model Name']} "
                        f"with AUC = {winner['AUC']:.4f}"
                    )

                    st.markdown("### Visual Comparison")
                    comparison_plot_df = rounded_comparison_df.set_index("ML Model Name")

                    fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
                    comparison_plot_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].plot(
                        kind="bar",
                        ax=ax_compare,
                    )
                    ax_compare.set_title("All Model Performance Comparison")
                    ax_compare.set_xlabel("ML Model")
                    ax_compare.set_ylabel("Score")
                    ax_compare.set_ylim(0, 1)
                    ax_compare.legend(loc="lower right")
                    ax_compare.tick_params(axis="x", rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig_compare)

                    comparison_csv = rounded_comparison_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download All Model Comparison as CSV",
                        data=comparison_csv,
                        file_name="all_model_comparison_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("No model comparison results were generated. Please check model files.")


    except Exception as e:
        st.error("Something went wrong while processing the file.")
        st.exception(e)

else:
    st.markdown(
        """
        <div class="section-card">
            <h3>👈 Upload a CSV file to start prediction</h3>
            <p>
                Use the sidebar to upload a CSV file. The app will load the selected model, generate predictions, calculate evaluation metrics, and display the confusion matrix and classification report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Expected CSV Format")
    st.markdown(
        """
        The uploaded CSV file should contain all input features and the target column `y`.

        ```text
        age, job, marital, education, default, balance, housing, loan,
        contact, day, month, duration, campaign, pdays, previous, poutcome, y
        ```

        Target column:

        ```text
        y = yes/no
        ```
        """
    )

st.markdown(
    """
    <div class="footer-note">
        Bank Marketing Classification Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
