from pathlib import Path
import json
import os

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import google.generativeai as genai


# =========================================================
# GEMINI API KEY
# =========================================================

# API key should be stored in environment variable:
# GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if API_KEY:
    genai.configure(api_key=API_KEY)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Based Cattle Disease Detection",
    page_icon="🐄",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "cattle_disease_model.keras"
LABELS_PATH = APP_DIR / "class_names.json"


# =========================================================
# DEFAULT CLASS NAMES
# =========================================================

DEFAULT_CLASS_NAMES = [
    "foot-and-mouth",
    "healthy",
    "lumpy"
]


DISPLAY_NAMES = {
    "foot-and-mouth": "Foot and Mouth Disease",
    "healthy": "Healthy",
    "lumpy": "Lumpy Skin Disease"
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_classifier():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    class_names = DEFAULT_CLASS_NAMES

    if LABELS_PATH.exists():
        class_names = json.loads(
            LABELS_PATH.read_text(encoding="utf-8")
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model, class_names


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # MobileNetV2 input size
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # Convert to numpy
    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Same preprocessing as MobileNetV2 training
    pixels = tf.keras.applications.mobilenet_v2.preprocess_input(
        pixels
    )

    # Add batch dimension
    return np.expand_dims(
        pixels,
        axis=0
    )


# =========================================================
# GEMINI INFORMATION
# =========================================================

def get_gemini_information(prediction, language):

    if not API_KEY:
        return (
            "Gemini API key is not configured.\n\n"
            "Please set GEMINI_API_KEY and restart the Streamlit app."
        )

    prompt = f"""
You are an educational assistant inside an AI Based Cattle Disease
Detection and Prevention System.

The cattle image classifier predicted:

{prediction}

Give short and simple information in {language}.

Include:

1. What this condition may mean
2. Common symptoms
3. General precautions
4. When to contact a veterinarian

Do not prescribe medicines.
Do not provide medicine dosage.
Do not claim that the AI prediction is a final veterinary diagnosis.

Keep the answer simple and useful for farmers.
"""

    try:

        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = gemini_model.generate_content(prompt)

        if response.text:
            return response.text

        return "No information was generated."

    except Exception:

        return (
            "Gemini is temporarily unavailable. "
            "The disease prediction itself is not affected "
            "by this Gemini error."
        )


# =========================================================
# PROJECT CHATBOT
# =========================================================

CHAT_TOPIC_WORDS = (
    "cattle",
    "cow",
    "cows",
    "calf",
    "livestock",
    "farmer",
    "farm",
    "foot and mouth",
    "foot-and-mouth",
    "fmd",
    "lumpy",
    "lumpy skin",
    "disease",
    "symptom",
    "symptoms",
    "prevention",
    "precaution",
    "veterinarian",
    "vet",
    "diagnosis",
    "healthy",
    "image",
    "upload",
    "camera",
    "prediction",
    "predict",
    "confidence",
    "classifier",
    "model",
    "tensorflow",
    "keras",
    "mobilenet",
    "project",
    "app",
    "cattle disease",
    "పశు",
    "ఆవు",
    "గేదె",
    "వ్యాధి",
    "లంపి"
)


def is_project_question(question):

    question = question.lower()

    return any(
        word in question
        for word in CHAT_TOPIC_WORDS
    )


def get_chat_response(question, language):

    # Block unrelated questions
    if not is_project_question(question):

        return (
            "I can only help with this cattle disease detection "
            "project: cattle health, Foot and Mouth Disease, "
            "Lumpy Skin Disease, image prediction, and how this app works."
        )

    # Check Gemini API
    if not API_KEY:

        return (
            "Chatbot is unavailable because GEMINI_API_KEY "
            "is not configured."
        )

    prompt = f"""
You are the assistant inside an AI Based Cattle Disease Detection
and Prevention System.

Answer only questions directly related to:

- Cattle health
- Foot and Mouth Disease
- Lumpy Skin Disease
- Healthy cattle
- Image prediction
- Disease prediction
- This Streamlit project
- TensorFlow
- Keras
- MobileNetV2

If the question is unrelated, reply:

"I can only answer cattle disease detection project questions."

Language: {language}

Keep the response short and clear.

Provide general educational information only.

Do not prescribe medicines.
Do not provide medicine dosages.
Advise consulting a veterinarian for diagnosis or treatment.

User question:
{question}
"""

    try:

        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = gemini_model.generate_content(
            prompt
        )

        return (
            response.text
            if response.text
            else "I could not generate a response."
        )

    except Exception:

        return (
            "Gemini chatbot is temporarily unavailable. "
            "Please try again in a few seconds."
        )


# =========================================================
# PROJECT CHATBOT UI
# =========================================================

def show_project_chatbot():

    st.divider()

    st.subheader("🐄 Cattle Project Chatbot")

    st.caption(
        "Ask only about cattle diseases, cattle health, "
        "predictions, or this project. Other topics are declined."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display old messages
    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.caption(
        "🎙️ Voice input is currently disabled because "
        "browser speech-to-text was unreliable."
    )

    question = st.chat_input(
        "Ask a cattle/project question"
    )

    if question:

        # User message
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # Assistant response
        answer = get_chat_response(
            question,
            st.session_state.language
        )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style="text-align: center;">
        🐄 AI Based Cattle Disease Detection<br>
        and Prevention System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.subheader("Description")

    st.write(
        "This project helps farmers identify possible cattle "
        "diseases from an image. Our deep learning model "
        "classifies cattle images as Healthy, Foot and Mouth "
        "Disease, or Lumpy Skin Disease. The system gives a "
        "prediction with confidence and helps farmers take "
        "early action by consulting a veterinarian."
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.subheader("Choose Language")

        language = st.selectbox(
            "Select language for disease information",
            ["English", "Telugu"],
            index=(
                0
                if st.session_state.language == "English"
                else 1
            )
        )

        st.session_state.language = language

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        if st.button(
            "Next ➜",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "prediction"

            st.rerun()


# =========================================================
# PREDICTION PAGE
# =========================================================

elif st.session_state.page == "prediction":

    col1, col2, col3 = st.columns(
        [1, 4, 1]
    )

    with col1:

        if st.button("← Back"):

            st.session_state.page = "home"

            st.rerun()

    with col2:

        st.markdown(
            """
            <h1 style="text-align: center;">
            🐄 AI Based Cattle Disease Detection<br>
            and Prevention System
            </h1>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        f"Selected language: "
        f"{st.session_state.language}"
    )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = None
    class_names = DEFAULT_CLASS_NAMES

    try:

        model, class_names = load_classifier()

    except FileNotFoundError:

        st.error(
            "Trained model not found. Please make sure "
            "cattle_disease_model.keras is in the same "
            "folder as app.py."
        )

    except Exception as error:

        st.error(
            "The TensorFlow model could not be loaded."
        )

        st.code(
            str(error)
        )


    # =====================================================
    # IMAGE SOURCE
    # =====================================================

    source = st.radio(
        "Choose Image Source",
        [
            "Upload Image",
            "Use Camera"
        ]
    )


    if source == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a cattle image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    else:

        uploaded_file = st.camera_input(
            "Take a cattle photo"
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Selected Cattle Image",
            width=350
        )


        if st.button(
            "🔍 Predict Disease",
            type="primary",
            disabled=model is None
        ):

            try:

                processed_image = preprocess_image(
                    image
                )

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                # Safety check
                if predicted_index >= len(class_names):

                    st.error(
                        "Model output classes do not match "
                        "the class names."
                    )

                else:

                    label = class_names[
                        predicted_index
                    ]

                    confidence = float(
                        predictions[
                            predicted_index
                        ]
                    )

                    st.session_state.prediction_result = {

                        "label": DISPLAY_NAMES.get(
                            label,
                            label
                        ),

                        "confidence": confidence,

                        "probabilities": {

                            DISPLAY_NAMES.get(
                                name,
                                name
                            ): float(score)

                            for name, score
                            in zip(
                                class_names,
                                predictions
                            )
                        }
                    }

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(error)
                )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    result = st.session_state.prediction_result

    if result:

        st.success(
            f"Prediction: {result['label']}"
        )

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}"
        )

        st.subheader(
            "Prediction Probability"
        )

        st.bar_chart(
            result["probabilities"]
        )


        # =================================================
        # GEMINI INFORMATION
        # =================================================

        st.subheader(
            "General Information"
        )

        st.write(
            get_gemini_information(
                result["label"],
                st.session_state.language
            )
        )

        st.warning(
            "This is an AI screening result only. "
            "Please consult a veterinarian for final "
            "diagnosis and treatment."
        )


    # =====================================================
    # CHATBOT
    # =====================================================

    show_project_chatbot()