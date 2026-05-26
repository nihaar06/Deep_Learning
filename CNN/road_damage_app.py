import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2

# Set page title
st.set_page_config(page_title="Road Damage Detector", page_icon="🚧")

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('road_damage_detector.keras')
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)
    return model, label_map

model, label_map = load_model_and_labels()

st.title("🚧 Road Damage Detection System")
st.write("Upload an image of a road to detect Potholes, Cracks, or Manholes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    prediction = model.predict(img_batch)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    result_label = label_map[str(class_idx)]
    
    # Output Results
    st.subheader(f"Prediction: {result_label}")
    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
    
    if confidence < 0.5:
        st.warning("Low confidence prediction. The model might be uncertain.")
