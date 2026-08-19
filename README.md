# AI Based Cattle Disease Detection and Prevention System

## Project Overview

This project uses Deep Learning to identify possible cattle diseases from an uploaded image.

The system classifies cattle images into:

- Healthy
- Foot and Mouth Disease
- Lumpy Skin Disease

The application also uses Gemini AI to provide general information about symptoms, precautions, and when to contact a veterinarian.

> Note: This project is an AI screening tool only. It does not replace a veterinarian's diagnosis.

---

## Features

- Upload cattle image
- Camera image input
- Disease prediction
- Confidence percentage
- Prediction probability chart
- English and Telugu language support
- Gemini AI general disease information
- Streamlit web application
- Sidebar with project and team information

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- scikit-learn
- Google Generative AI
- tf-keras

---

## Model Used

This project uses **MobileNetV2**, a lightweight Convolutional Neural Network model.

MobileNetV2 is used because:

- It is fast.
- It is suitable for image classification.
- It uses transfer learning.
- It works well with limited resources.

The input image is resized to:

```text
224 × 224 pixels