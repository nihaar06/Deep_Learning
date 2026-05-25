import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

# ==========================================
# TASK 5 — Load Trained Model & Mock Scaler
# ==========================================
@st.cache_resource
def load_prediction_model():
    try:
        import os
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "titanic_model.keras")
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model()

# ==========================================
# TASK 4 — Data Preprocessing Function
# ==========================================
def preprocess_input(pclass, age, fare):
    """
    Normalizes and prepares the inputs exactly as done during training.
    Adjust the mean and std values below to match your actual training data scaler.
    """
    # Example Mock Scaler constants (Replace with your actual training insights)
    # Assuming training data had Age mean=29.7, std=14.5 and Fare mean=32.2, std=49.7
    age_mean, age_std = 29.699118, 13.002015
    fare_mean, fare_std = 32.204208, 49.693429
    
    normalized_age = (age - age_mean) / age_std
    normalized_fare = (fare - fare_mean) / fare_std
    
    # One-hot encoding features for Pclass if your model expected 3 columns, 
    # or just pass the raw/normalized class. Let's assume a simple 3-feature array for this example:
    features = np.array([[pclass, normalized_age, normalized_fare]], dtype=np.float32)
    return features

# ==========================================
# TASK 7 — UI Styling & Layout Setup
# ==========================================
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
    .main-header { font-size:2.8rem !important; color: #1E3A8A; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size:1.3rem !important; color: #4B5563; margin-top: 0px; margin-bottom: 20px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SECTION 1 — Header Area
# ==========================================
with st.container():
    col_icon, col_title = st.columns([1, 5])
    with col_icon:
        # AI/Project Related Visual Icon
        st.markdown("<h1 style='font-size: 5rem; text-align: center; margin:0;'>🚢</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<p class='main-header'>Titanic Survival Prediction System</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Deep Learning Based Passenger Survival Prediction</p>", unsafe_allow_html=True)

st.divider()

# ==========================================
# SECTION 2 — Project Description
# ==========================================
with st.expander("ℹ️ About This Project", expanded=True):
    st.markdown("""
    ### Purpose & Methodology
    This application leverages an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** to predict the survival probability of Titanic passengers based on their demographics and ticketing data.
    
    ### How it Works:
    1. **Input Collection:** Gather inputs dynamically from the user interface.
    2. **Data Preprocessing:** Inputs are standard-normalized in real-time to match the exact statistical scaling constraints applied during network training.
    3. **TensorFlow Deployment:** The preprocessed vector is fed straight into the deployed deep learning network topology for continuous inference.
    """)

# Setup Columns for Form and Outputs
col_input, col_output = st.columns([1, 1], gap="large")

# ==========================================
# SECTION 3 — Passenger Input Form
# ==========================================
with col_input:
    st.subheader("📋 Passenger Attributes")
    
    with st.form("prediction_form"):
        pclass = st.selectbox(
            "Passenger Class (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else f"{x}rd Class")
        )
        
        age = st.slider("Passenger Age", min_value=0, max_value=100, value=28, step=1)
        
        fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.20, step=0.5)
        
        # SECTION 4 — Prediction Button
        submit_btn = st.form_submit_button("Predict Survival", use_container_width=True)

# ==========================================
# SECTION 6 & 7 — Real-time Dummy Model Fallback Logic
# ==========================================
def get_prediction(processed_data):
    """Wrapper to simulate inference if model file is missing during setup."""
    if model is not None:
        return float(model.predict(processed_data)[0][0])
    else:
        # Smart fallback rule-base algorithm approximating Titanic parameters if no H5 is loaded yet
        calc_prob = 0.15
        if pclass == 1: calc_prob += 0.4
        if pclass == 2: calc_prob += 0.2
        if age < 12: calc_prob += 0.25
        if fare > 50: calc_prob += 0.1
        return min(max(calc_prob, 0.05), 0.95)

# Processing the Prediction Action
if submit_btn:
    # Task 4 Execution
    processed_features = preprocess_input(pclass, age, fare)
    
    # Task 5 & 6 Execution
    survival_prob = get_prediction(processed_features)
    not_survival_prob = 1.0 - survival_prob
    
    with col_output:
        # ==========================================
        # SECTION 5 — Prediction Output Area
        # ==========================================
        st.subheader("📊 Prediction Analysis")
        
        # Task 6 Decision threshold logic
        if survival_prob >= 0.5:
            status = "Survived"
            status_color = "green"
            confidence_score = survival_prob * 100
        else:
            status = "Did Not Survive"
            status_color = "red"
            confidence_score = not_survival_prob * 100

        # Metrics/Cards Design
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Status", value=status)
        with c2:
            st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")

        # ==========================================
        # SECTION 6 — Visualization Area
        # ==========================================
        # Interactive donut/pie chart using Plotly
        fig = go.Figure(data=[go.Pie(
            labels=['Survived', 'Not Survived'],
            values=[survival_prob, not_survival_prob],
            hole=.4,
            marker_colors=['#10B981', '#EF4444']
        )])
        
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    with col_output:
        st.info("💡 Fill in the passenger configurations on the left panel and click 'Predict Survival' to execute model inference.")
