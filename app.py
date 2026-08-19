from pathlib import Path
import json

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import google.generativeai as genai

try:
    import tf_keras as keras_loader
except ImportError:
    keras_loader = tf.keras


# =========================================================
# PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="AI Based Cattle Disease Detection",
    page_icon="🐄",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cattle_disease_model.keras"
LABELS_PATH = APP_DIR / "class_names.json"

DEFAULT_CLASS_NAMES = [
    "foot-and-mouth",
    "healthy",
    "lumpy",
    "not-cattle"
]

DISPLAY_NAMES = {
    "foot-and-mouth": "Foot and Mouth Disease",
    "healthy": "Healthy",
    "lumpy": "Lumpy Skin Disease",
    "not-cattle": "Not a Cattle Image"
}


# =========================================================
# GEMINI API KEY FROM STREAMLIT SECRETS
# =========================================================
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if API_KEY:
    genai.configure(api_key=API_KEY)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Artificial Intelligence Career for Women (AICW)")

st.sidebar.markdown("### VSM College of Engineering")

st.sidebar.markdown("### Project Guide")
st.sidebar.write("Mr. Abdul Aziz Md")

st.sidebar.markdown("### Team Members")
st.sidebar.write("**P. V. S. D. Akhila** — Team Lead")
st.sidebar.write("R. Sireesha — Team Member")
st.sidebar.write("P. S. B. Anuradha Mutyaveni — Team Member")
st.sidebar.write("K. Ramya — Team Member")


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

    model = keras_loader.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output and class_names.json do not match. "
            "Retrain the model with all four classes."
        )

    return model, class_names


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    pixels = tf.keras.applications.mobilenet_v2.preprocess_input(
        pixels
    )

    return np.expand_dims(
        pixels,
        axis=0
    )


# =========================================================
# GEMINI INFORMATION
# =========================================================
def get_gemini_information(prediction, language):

    if not API_KEY:
        return "Gemini API key is not configured."

    prompt = f"""
The cattle image classifier predicted: {prediction}.

Give short and simple information in {language}:
1. What this condition may mean
2. Common symptoms
3. General precautions
4. When to contact a veterinarian

Do not give medicine dosage.
Do not say this is a final veterinary diagnosis.
"""

    try:
        gemini_model = genai.GenerativeModel(
            "gemini-3.6-flash"
        )

        response = gemini_model.generate_content(prompt)

        return response.text

    except Exception as error:
        return f"Gemini information error: {error}"


# =========================================================
# PAGE NAVIGATION
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "language" not in st.session_state:
    st.session_state.language = "English"


# =========================================================
# HOME PAGE
# =========================================================
if st.session_state.page == "home":

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style="text-align: center;">
        🐄 AI Based Cattle Disease Detection<br>
        and Prevention System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Description")

    st.write(
        "This project helps farmers identify possible cattle diseases from "
        "an image. Our deep learning model classifies cattle images as "
        "Healthy, Foot and Mouth Disease, or Lumpy Skin Disease. If the "
        "image is not a cattle image, the system displays a warning message."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.subheader("Choose Language")

        language = st.selectbox(
            "Select language for disease information",
            ["English", "Telugu"],
            index=0 if st.session_state.language == "English" else 1
        )

        st.session_state.language = language

        st.markdown("<br>", unsafe_allow_html=True)

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

    col1, col2, col3 = st.columns([1, 4, 1])

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
        f"Selected language: {st.session_state.language}"
    )

    try:
        model, class_names = load_classifier()

    except Exception as error:
        st.error("The model could not be loaded.")
        st.code(str(error))
        st.stop()

    source = st.radio(
        "Choose Image Source",
        ["Upload Image", "Use Camera"]
    )

    if source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a cattle image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input(
            "Take a cattle photo"
        )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Selected Image",
            width=350
        )

        if st.button(
            "🔍 Predict Disease",
            type="primary"
        ):

            processed_image = preprocess_image(image)

            predictions = model.predict(
                processed_image,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            label = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            )

            if label == "not-cattle":
                st.error(
                    "Please upload cattle images only."
                )
                st.stop()

            display_label = DISPLAY_NAMES.get(
                label,
                label
            )

            st.success(
                f"Prediction: {display_label}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

            st.subheader("Prediction Probability")

            st.bar_chart({
                DISPLAY_NAMES.get(name, name): float(score)
                for name, score in zip(
                    class_names,
                    predictions
                )
            })

            st.subheader("General Information")

            st.write(
                get_gemini_information(
                    display_label,
                    st.session_state.language
                )
            )

            st.warning(
                "This is an AI screening result only. "
                "Please consult a veterinarian for final diagnosis and treatment."
            )