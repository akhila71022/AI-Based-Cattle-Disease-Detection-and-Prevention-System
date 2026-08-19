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


# =========================================================
# FILE PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "cattle_disease_model.keras"
LABELS_PATH = APP_DIR / "class_names.json"


# =========================================================
# CLASS NAMES
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
# GEMINI API KEY
# =========================================================

API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if API_KEY:
    genai.configure(api_key=API_KEY)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Artificial Intelligence Career for Women (AICW)"
)

st.sidebar.markdown(
    "### VSM College of Engineering"
)

st.sidebar.markdown(
    "### Project Guide"
)

st.sidebar.write(
    "Mr. Abdul Aziz Md"
)

st.sidebar.markdown(
    "### Team Members"
)

st.sidebar.write(
    "**P. V. S. D. Akhila** — Team Lead"
)

st.sidebar.write(
    "R. Sireesha — Team Member"
)

st.sidebar.write(
    "P. S. B. Anuradha Mutyaveni — Team Member"
)

st.sidebar.write(
    "K. Ramya — Team Member"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_classifier():

    # Check model file
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Default classes
    class_names = DEFAULT_CLASS_NAMES

    # Read class_names.json if available
    if LABELS_PATH.exists():

        try:
            class_names = json.loads(
                LABELS_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            class_names = DEFAULT_CLASS_NAMES

    # Make sure class names are strings
    class_names = [
        str(name).strip().lower()
        for name in class_names
    ]

    # Load model
    model = keras_loader.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Get model output classes
    model_output_classes = model.output_shape[-1]

    # Check number of classes
    if model_output_classes != len(class_names):

        # If JSON is wrong but model is 3-class,
        # automatically use the correct 3 classes.
        if model_output_classes == 3:

            class_names = [
                "foot-and-mouth",
                "healthy",
                "lumpy"
            ]

        else:

            raise ValueError(
                f"Model has {model_output_classes} output classes, "
                f"but class_names.json has {len(class_names)} classes."
            )

    return model, class_names


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy array
    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    pixels = tf.keras.applications.mobilenet_v2.preprocess_input(
        pixels
    )

    # Add batch dimension
    pixels = np.expand_dims(
        pixels,
        axis=0
    )

    return pixels


# =========================================================
# GEMINI INFORMATION
# =========================================================

def get_gemini_information(
    prediction,
    language
):

    if not API_KEY:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in Streamlit Secrets."
        )

    prompt = f"""
The cattle image classifier predicted:
{prediction}

Give short and simple information in {language}.

Include:
1. What this condition may mean
2. Common symptoms
3. General precautions
4. When to contact a veterinarian

Do not give medicine dosage.
Do not provide a final veterinary diagnosis.
Use simple language.
"""

    try:

        # Gemini model
        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = gemini_model.generate_content(
            prompt
        )

        return response.text

    except Exception as error:

        return (
            f"Gemini information error: {error}"
        )


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:

    st.session_state.page = "home"


if "language" not in st.session_state:

    st.session_state.language = "English"


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
        "This project helps farmers identify possible "
        "cattle diseases from an image. Our deep learning "
        "model classifies cattle images as Healthy, "
        "Foot and Mouth Disease, or Lumpy Skin Disease."
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.subheader(
            "Choose Language"
        )

        language = st.selectbox(
            "Select language for disease information",
            [
                "English",
                "Telugu"
            ],
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

    # Back button
    with col1:

        if st.button("← Back"):

            st.session_state.page = "home"

            st.rerun()

    # Title
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

    try:

        model, class_names = load_classifier()

    except Exception as error:

        st.error(
            "The model could not be loaded."
        )

        st.code(
            str(error)
        )

        st.stop()


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


    # =====================================================
    # UPLOAD IMAGE
    # =====================================================

    if source == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a cattle image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


    # =====================================================
    # CAMERA
    # =====================================================

    else:

        uploaded_file = st.camera_input(
            "Take a cattle photo"
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            )

        except Exception:

            st.error(
                "Unable to read the selected image."
            )

            st.stop()


        # Display image
        st.image(
            image,
            caption="Selected Image",
            width=350
        )


        # Predict button
        if st.button(
            "🔍 Predict Disease",
            type="primary"
        ):

            try:

                # Preprocess
                processed_image = preprocess_image(
                    image
                )

                # Prediction
                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                # Find highest probability
                predicted_index = int(
                    np.argmax(predictions)
                )

                # Get label
                label = class_names[
                    predicted_index
                ]

                # Confidence
                confidence = float(
                    predictions[
                        predicted_index
                    ]
                )


                # =================================================
                # DISPLAY RESULT
                # =================================================

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


                # =================================================
                # PROBABILITY
                # =================================================

                st.subheader(
                    "Prediction Probability"
                )

                probability_data = {
                    DISPLAY_NAMES.get(
                        name,
                        name
                    ): float(score)

                    for name, score in zip(
                        class_names,
                        predictions
                    )
                }

                st.bar_chart(
                    probability_data
                )


                # =================================================
                # GEMINI INFORMATION
                # =================================================

                st.subheader(
                    "General Information"
                )

                information = get_gemini_information(
                    display_label,
                    st.session_state.language
                )

                st.write(
                    information
                )


                # =================================================
                # WARNING
                # =================================================

                st.warning(
                    "This is an AI screening result only. "
                    "Please consult a veterinarian for "
                    "final diagnosis and treatment."
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(error)
                )
