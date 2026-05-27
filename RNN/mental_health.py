import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    h1 {
        color: #38bdf8;
        text-align: center;
        font-size: 42px;
    }

    h2, h3 {
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 20px;
        margin-bottom: 30px;
    }

    .box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }

    .prediction {
        font-size: 28px;
        color: #22c55e;
        font-weight: bold;
    }

    .confidence {
        font-size: 22px;
        color: #facc15;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model("simple_rnn_sentiment_model.keras")

# =====================================================
# LOAD TOKENIZER
# =====================================================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =====================================================
# LOAD LABEL ENCODER
# =====================================================
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 50

# =====================================================
# HEADER SECTION
# =====================================================
st.markdown(
    """
    <h1>AI-Based Mental Health Sentiment Monitoring System</h1>
    <p class='subtitle'>Emotion Detection using Simple Recurrent Neural Networks</p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# ABOUT PROJECT
# =====================================================
st.markdown("<div class='box'>", unsafe_allow_html=True)
st.header("📘 About the Project")

st.write(
    """
    Emotional AI is becoming increasingly important in healthcare, education,
    and digital communication systems. This project uses Natural Language
    Processing (NLP) and Deep Learning techniques to detect emotions from text.

    Recurrent Neural Networks (RNNs) are highly effective for sequence learning
    because they process text word-by-word while remembering contextual
    information from previous words.

    Applications include:

    - Mental health monitoring
    - Emotion-aware chatbots
    - Social media sentiment analysis
    - Human-computer interaction systems
    """
)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# SAMPLE TEXTS
# =====================================================
sample_texts = [
    "I feel stressed and overwhelmed with work.",
    "I am extremely happy and excited today!",
    "I feel lonely and emotionally exhausted.",
    "Everything is going great in my life.",
    "I am anxious about my future."
]

# =====================================================
# USER INPUT AREA
# =====================================================
st.markdown("<div class='box'>", unsafe_allow_html=True)
st.header("✍️ Enter Your Thoughts")

st.write("### Sample Sentences")
for text in sample_texts:
    st.write(f"• {text}")

user_input = st.text_area(
    "Enter Text",
    placeholder="Enter your thoughts or feelings here...",
    height=180
)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    return padded

# =====================================================
# GUIDANCE FUNCTION
# =====================================================
def get_guidance(emotion):

    guidance = {
        "sadness": (
            "Take a short break and talk with someone you trust.",
            "Go for a short walk or listen to calming music."
        ),

        "anxiety": (
            "Focus on deep breathing and stay present.",
            "Try meditation or journaling for 10 minutes."
        ),

        "anger": (
            "Pause before reacting emotionally.",
            "Try physical exercise to release stress."
        ),

        "joy": (
            "Keep spreading positivity around you.",
            "Celebrate your achievements today."
        ),

        "fear": (
            "Remember that difficult moments pass with time.",
            "Talk with supportive friends or family."
        )
    }

    emotion = emotion.lower()

    if emotion in guidance:
        return guidance[emotion]

    return (
        "Stay positive and take care of yourself.",
        "Maintain healthy habits and regular sleep."
    )

# =====================================================
# PREDICTION BUTTON
# =====================================================
if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        processed_text = preprocess_text(user_input)

        prediction = model.predict(processed_text)

        predicted_index = np.argmax(prediction)

        predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

        confidence = np.max(prediction) * 100

        # =============================================
        # PREDICTION OUTPUT
        # =============================================
        st.markdown("<div class='box'>", unsafe_allow_html=True)

        st.header("🧠 Prediction Result")

        st.markdown(
            f"<p class='prediction'>Emotion Detected: {predicted_emotion}</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p class='confidence'>Confidence: {confidence:.2f}%</p>",
            unsafe_allow_html=True
        )

        emotional_status = "Stable"

        if confidence > 85:
            emotional_status = "High Emotional Intensity"

        st.write(f"### Emotional Status: {emotional_status}")

        st.markdown("</div>", unsafe_allow_html=True)

        # =============================================
        # VISUALIZATION SECTION
        # =============================================
        st.markdown("<div class='box'>", unsafe_allow_html=True)

        st.header("📊 Sentiment Confidence Graph")

        emotions = label_encoder.classes_
        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(emotions, probabilities)

        ax.set_xlabel("Emotions")
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Probability Distribution")

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # =============================================
        # GUIDANCE SECTION
        # =============================================
        message, activity = get_guidance(predicted_emotion)

        st.markdown("<div class='box'>", unsafe_allow_html=True)

        st.header("💡 Emotional Wellness Guidance")

        st.success(message)

        st.info(f"Suggested Positive Activity: {activity}")

        st.write(
            """
            ### Wellness Tips

            - Stay hydrated
            - Sleep properly
            - Exercise regularly
            - Practice mindfulness
            - Stay connected with supportive people
            """
        )

        st.markdown("</div>", unsafe_allow_html=True)
