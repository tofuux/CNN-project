import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("outputs/best_model.keras")

# Preprocess function
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# UI
st.title("🧠 Handwriting Recognition AI")

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    processed = preprocess_image(img)
    prediction = model.predict(processed)
    result = np.argmax(prediction)

    st.success(f"Predicted Digit: {result}")