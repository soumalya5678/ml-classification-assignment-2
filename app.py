import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Machine Learning Classification App")
st.write("UCI Breast Cancer Wisconsin (Diagnostic) dataset")

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

model_name = st.selectbox("Select a classification model", list(model_files.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.info("Use the supplied test_data.csv for evaluation. It contains the target column.")

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if "target" not in df.columns:
        st.error("The CSV must contain a 'target' column.")
        st.stop()

    model = joblib.load(model_files[model_name])
    X = df.drop(columns=["target"])
    y = df["target"]

    # Keep only the feature columns used during training and in the same order.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "named_steps") and "model" in model.named_steps and hasattr(model.named_steps["model"], "feature_names_in_"):
        expected = list(model.named_steps["model"].feature_names_in_)
    else:
        expected = list(X.columns)

    missing = [c for c in expected if c not in X.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}")
        st.stop()

    X = X[expected]
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
    col3.metric("AUC", f"{roc_auc_score(y, prob):.4f}")

    st.subheader("Evaluation Metrics")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Score": [
            accuracy_score(y, pred),
            roc_auc_score(y, prob),
            precision_score(y, pred),
            recall_score(y, pred),
            f1_score(y, pred),
            matthews_corrcoef(y, pred)
        ]
    })
    metrics["Score"] = metrics["Score"].round(4)
    st.table(metrics)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                              columns=["Predicted 0", "Predicted 1"]))

    st.subheader("Predictions")
    result = df.copy()
    result["prediction"] = pred
    result["probability_class_1"] = prob.round(4)
    st.dataframe(result, use_container_width=True)
