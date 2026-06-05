
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)
from pathlib import Path

st.write("Current directory:", Path.cwd())

st.write(
    list(Path(__file__).parent.iterdir())
)
from pathlib import Path
from tensorflow.keras.models import load_model

MODEL_PATH = Path(__file__).parent / "fraud_lstm_attention.keras"

@st.cache_resource
def load_fraud_model():
    return load_model(MODEL_PATH)

model = load_fraud_model()

st.title("💳 Fraud Detection System")
st.markdown("LSTM + MultiHeadAttention Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:

        X = df.values.astype(np.float32)

        SEQ_LEN = 5

        X_seq = []

        for i in range(len(X)-SEQ_LEN):
            X_seq.append(X[i:i+SEQ_LEN])

        X_seq = np.array(X_seq)

        probs = model.predict(X_seq)

        results = pd.DataFrame({
            "Fraud_Probability": probs.flatten()
        })

        results["Risk"] = np.where(
            results["Fraud_Probability"] > 0.5,
            "High Risk",
            "Low Risk"
        )

        st.subheader("Fraud Predictions")
        st.dataframe(results)

        st.subheader("High Risk Transactions")

        high_risk = results[
            results["Fraud_Probability"] > 0.5
        ]

        st.dataframe(high_risk)

        st.subheader("Fraud Probability Distribution")

        st.bar_chart(
            results["Fraud_Probability"]
        )

    except Exception as e:
        st.error(str(e))
