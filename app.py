import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = "intrusion_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Streamlit UI
st.title("🔊 Intrusion Detection via Sound")
st.write("Upload an audio file (.wav) to detect if there is an intrusion.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features(file_path)
    prediction = model.predict(np.expand_dims(features, axis=0))

    result = "🚨 Intrusion Detected!" if prediction[0] > 0.5 else "✅ No Intrusion"
    st.write(f"**Prediction:** {result}")
