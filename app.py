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

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
    
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_index]
    return predicted_label

st.title("Teeth Condition Classification")

uploaded_file = st.file_uploader("Choose an image representing one of these categories: CaS, CoS, Gum, MC, OC, OLP or OT", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    prediction = predict(image)

    st.write(f"Prediction: {prediction}")
