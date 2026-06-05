import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Dense
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("💳 Fraud Detection System")
st.markdown("LSTM + MultiHeadAttention Fraud Detection")

# ==========================
# MODEL PARAMETERS
# ==========================

SEQ_LEN = 5
NUM_FEATURES = 29

# ==========================
# REBUILD MODEL ARCHITECTURE
# ==========================

@st.cache_resource
def load_fraud_model():

    inputs = Input(shape=(SEQ_LEN, NUM_FEATURES))

    x = LSTM(
        64,
        return_sequences=True
    )(inputs)

    x = MultiHeadAttention(
        num_heads=8,
        key_dim=64
    )(x, x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.load_weights(
        "fraud_weights.weights.h5"
    )

    return model


model = load_fraud_model()

st.success("✅ Model Loaded Successfully")

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

# ==========================
# PROCESS FILE
# ==========================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        st.write("Original Shape:", df.shape)

        # ==========================
        # DROP UNUSED COLUMNS
        # ==========================

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        if "Time" in df.columns:
            df = df.drop("Time", axis=1)

        st.write("After Preprocessing:", df.shape)

        # ==========================
        # FEATURE CHECK
        # ==========================

        if df.shape[1] != NUM_FEATURES:

            st.error(
                f"""
                Model expects {NUM_FEATURES} features.

                Uploaded file contains {df.shape[1]} features.
                """
            )

            st.stop()

        X = df.values.astype(np.float32)

        # ==========================
        # SEQUENCE CHECK
        # ==========================

        if len(X) <= SEQ_LEN:

            st.error(
                f"""
                Need more than {SEQ_LEN} rows.

                Uploaded rows = {len(X)}
                """
            )

            st.stop()

        # ==========================
        # CREATE SEQUENCES
        # ==========================

        X_seq = []

        for i in range(len(X) - SEQ_LEN):

            X_seq.append(
                X[i:i + SEQ_LEN]
            )

        X_seq = np.array(X_seq)

        st.write("Sequence Shape:", X_seq.shape)

        # ==========================
        # PREDICT
        # ==========================

        with st.spinner("Generating Predictions..."):

            probs = model.predict(
                X_seq,
                verbose=0
            )

        # ==========================
        # RESULTS
        # ==========================

        results = pd.DataFrame()

        results["Fraud Probability"] = probs.flatten()

        results["Risk"] = np.where(
            results["Fraud Probability"] > 0.5,
            "High Risk",
            "Low Risk"
        )

        st.subheader("Fraud Predictions")

        st.dataframe(results)

        # ==========================
        # HIGH RISK TRANSACTIONS
        # ==========================

        high_risk = results[
            results["Fraud Probability"] > 0.5
        ]

        st.subheader("🚨 High Risk Transactions")

        if len(high_risk) > 0:

            st.dataframe(high_risk)

        else:

            st.success(
                "No High Risk Transactions Detected"
            )

        # ==========================
        # METRICS
        # ==========================

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

        # ==========================
        # CHART
        # ==========================

        st.subheader("Fraud Probability Distribution")

        st.line_chart(
            results["Fraud Probability"]
        )

    except Exception as e:

        import traceback

        st.error("Prediction Error")

        st.code(
            traceback.format_exc()
        )
