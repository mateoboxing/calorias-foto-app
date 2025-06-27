import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.applications.MobileNetV2(weights='imagenet')
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

def predict_food(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]
    return decoded

def estimate_calories(predictions):
    calorie_table = {
        'pizza': 285,
        'hamburger': 250,
        'apple': 95,
        'banana': 105,
        'hotdog': 150,
        'broccoli': 55,
        'carrot': 25,
        'cake': 300,
        'rice': 206,
        'chicken': 239
    }
    
    for _, label, prob in predictions:
        if label.lower() in calorie_table:
            return f"Detectado: {label} ({prob*100:.1f}%) | Estimado: {calorie_table[label.lower()]} kcal"
    return "No se detectó alimento conocido para estimar calorías."

st.title("Calorías con Foto 🍽️")
st.write("Sube una foto de tu comida para estimar calorías de forma rápida.")

uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    st.write("Analizando imagen...")
    
    predictions = predict_food(image)
    result = estimate_calories(predictions)
    
    st.write(result)
    st.write("Predicciones detalladas:")
    for _, label, prob in predictions:
        st.write(f"{label}: {prob*100:.2f}%")
