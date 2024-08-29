import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

MODEL_URL = "https://github.com/ZumZuo/Computer-Vision_Cellula-Tech/releases/download/finetuned/resnet50v2_pretuned_final.keras"
MODEL_PATH = "resnet50v2_pretuned_final.keras"

@st.cache(allow_output_mutation=True)
def download_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = download_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust based on your model's input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit interface
st.title("Teeth Condition Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    prediction = predict(image)

    st.write(f"Prediction: {prediction}")
