import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI-Based Road Damage Detection System",
    page_icon="🚧",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f2937;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.upload-box {
    padding: 20px;
    border: 2px dashed #4B9CD3;
    border-radius: 10px;
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(
    "road_damage_detector.keras",
    compile=False,
    safe_mode=False
)


# =====================================================
# LOAD LABEL MAP
# =====================================================

with open("label_map.json", "r") as f:
    label_mapping = json.load(f)

class_names = label_mapping


# =====================================================
# SECTION 1 — HEADER
# =====================================================

st.markdown("""
# 🚧 AI-Based Road Damage Detection System
### Smart City Infrastructure Monitoring using CNN
""")

st.write("---")


# =====================================================
# SECTION 2 — ABOUT PROJECT
# =====================================================

with st.container():

    st.header("📘 About the Project")

    st.write("""
Road monitoring is essential for maintaining safe transportation systems,
preventing accidents, and improving urban infrastructure management.

Traditional road inspections are slow and expensive.
Artificial Intelligence enables automated road analysis using computer vision.

This project uses Convolutional Neural Networks (CNNs) to analyze road surface
images and detect different types of road damage such as:

- Potholes
- Cracks
- Manholes

CNNs are powerful deep learning models that automatically learn visual patterns,
textures, and features directly from images.

### Industry Applications
- Smart city infrastructure monitoring
- Highway maintenance automation
- Municipal road inspection systems
- Accident prevention systems
- AI-powered traffic safety monitoring
""")

st.write("---")


# =====================================================
# SECTION 3 — UPLOAD AREA
# =====================================================

st.header("📤 Upload Area")

st.markdown("""
<div class="upload-box">
Upload a road image for AI-based inspection.
Supported formats: JPG, JPEG, PNG
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

st.write("---")


# =====================================================
# SECTION 4 — IMAGE PREVIEW
# =====================================================

if uploaded_file is not None:

    image_file = Image.open(uploaded_file)

    st.header("🖼 Uploaded Image Preview")

    preview_col, prediction_col = st.columns([1,1])

    with preview_col:

        st.image(
            image_file,
            caption="Uploaded Road Image",
            use_container_width=True
        )

    # =================================================
    # IMAGE PREPROCESSING
    # =================================================

    resized_img = image_file.resize((224,224))

    img_array = image.img_to_array(resized_img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # =================================================
    # MODEL PREDICTION
    # =================================================

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[str(predicted_index)]

    confidence = float(np.max(prediction) * 100)

    # =================================================
    # SEVERITY LEVEL
    # =================================================

    if predicted_class.lower() == "pothole":
        severity = "High"

    elif predicted_class.lower() == "crack":
        severity = "Medium"

    else:
        severity = "Low"

    # =================================================
    # SECTION 5 — PREDICTION AREA
    # =================================================

    with prediction_col:

        st.header("🔍 Prediction Area")

        st.success(
            f"Prediction: {predicted_class} Detected"
        )

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with metric2:
            st.metric(
                "Severity",
                severity
            )

        if severity == "High":
            st.error("⚠ High-risk road condition detected")

        elif severity == "Medium":
            st.warning("⚠ Moderate road damage detected")

        else:
            st.success("✔ Low-risk condition detected")

    st.write("---")

    # =================================================
    # SECTION 6 — VISUALIZATION AREA
    # =================================================

    st.header("📊 Visualization Area")

    probabilities = prediction[0]

    labels = list(class_names.values())

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        labels,
        probabilities * 100
    )

    ax.set_xlabel("Damage Categories")

    ax.set_ylabel("Confidence (%)")

    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    st.subheader("Probability Details")

    for label, prob in zip(labels, probabilities):

        st.progress(float(prob))

        st.write(
            f"{label}: {prob*100:.2f}%"
        )

    st.write("---")

    # =================================================
    # SECTION 7 — RECOMMENDATIONS
    # =====================================================

    st.header("🛠 Recommendations")

    if severity == "High":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.

This damage may cause vehicle accidents
and requires urgent repair.
""")

    elif severity == "Medium":

        st.warning("""
Repair work should be scheduled soon.

Road condition may worsen over time.

Continuous monitoring recommended.
""")

    else:

        st.success("""
Low-risk road condition detected.

Routine maintenance inspection recommended.
""")

else:

    st.info(
        "Upload a road image to begin AI-based road damage detection."
    )