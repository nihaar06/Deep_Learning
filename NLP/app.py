import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from pathlib import Path
import traceback

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("💳 Fraud Detection System")
st.markdown("LSTM + MultiHeadAttention Fraud Detection")

# =====================================
# MODEL CONFIG
# =====================================

SEQ_LEN = 5
EXPECTED_FEATURES = 29

# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = Path(__file__).parent / "fraud_lstm_attention.keras"

@st.cache_resource
def load_fraud_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_fraud_model()
    st.success("✅ Model Loaded Successfully")

except Exception:
    st.error("❌ Failed to load model")
    st.code(traceback.format_exc())
    st.stop()

# =====================================
# FILE UPLOADER
# =====================================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # =====================================
        # READ DATA
        # =====================================

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        st.write("Original Shape:", df.shape)

        # =====================================
        # REMOVE UNUSED COLUMNS
        # =====================================

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        if "Time" in df.columns:
            df = df.drop("Time", axis=1)

        st.write("After Preprocessing:", df.shape)

        # =====================================
        # FEATURE CHECK
        # =====================================

        if df.shape[1] != EXPECTED_FEATURES:

            st.error(
                f"""
                Model expects {EXPECTED_FEATURES} features.

                Uploaded CSV contains {df.shape[1]} features.
                """
            )

            st.stop()

        # =====================================
        # CONVERT TO NUMPY
        # =====================================

        X = df.values.astype(np.float32)

        st.write("Input Shape:", X.shape)

        # =====================================
        # SEQUENCE CHECK
        # =====================================

        if len(X) <= SEQ_LEN:

            st.error(
                f"""
                Need more than {SEQ_LEN} rows.

                Uploaded rows = {len(X)}
                """
            )

            st.stop()

        # =====================================
        # CREATE SEQUENCES
        # =====================================

        X_seq = []

        for i in range(len(X) - SEQ_LEN):
            X_seq.append(X[i:i + SEQ_LEN])

        X_seq = np.array(X_seq)

        st.write("Sequence Shape:", X_seq.shape)

        # =====================================
        # PREDICTION
        # =====================================

        with st.spinner("Predicting Fraud Probability..."):

            probs = model.predict(
                X_seq,
                verbose=0
            )

        # =====================================
        # RESULTS
        # =====================================

        results = pd.DataFrame({
            "Fraud Probability": probs.flatten()
        })

        results["Risk"] = np.where(
            results["Fraud Probability"] > 0.5,
            "High Risk",
            "Low Risk"
        )

        # =====================================
        # DISPLAY RESULTS
        # =====================================

        st.subheader("Fraud Predictions")

        st.dataframe(results)

        # =====================================
        # HIGH RISK
        # =====================================

        high_risk = results[
            results["Fraud Probability"] > 0.5
        ]

        st.subheader("🚨 High Risk Transactions")

        if len(high_risk) > 0:
            st.dataframe(high_risk)
        else:
            st.success("No High Risk Transactions Found")

        # =====================================
        # METRICS
        # =====================================

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Predictions",
                len(results)
            )

        with col2:
            st.metric(
                "High Risk Count",
                len(high_risk)
            )

        # =====================================
        # CHART
        # =====================================

        st.subheader("Fraud Probability Distribution")

        st.line_chart(
            results["Fraud Probability"]
        )

        # =====================================
        # TOP RISKY TRANSACTIONS
        # =====================================

        st.subheader("Top 10 Most Suspicious Transactions")

        top10 = results.sort_values(
            by="Fraud Probability",
            ascending=False
        ).head(10)

        st.dataframe(top10)

    except Exception:

        st.error("Prediction Failed")

        st.code(
            traceback.format_exc()
        )
