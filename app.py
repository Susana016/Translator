import streamlit as st
import numpy as np
import librosa
import pickle
import os
import csv
from tensorflow import keras
import tempfile

# Page configuration
st.set_page_config(page_title="English to Spanish Speech Translator", page_icon="🎤")

st.title("🎤 English to Spanish Speech Translator")
st.write("Record your voice saying an English word and get the Spanish translation!")

# Load the dictionary
@st.cache_resource
def load_dictionary():
    """Load the English-Spanish dictionary from CSV"""
    dictionary = {}
    try:
        with open('dictionary.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                english = row['English'].strip().lower()
                spanish = row['Spanish'].strip()
                dictionary[english] = spanish
        return dictionary
    except Exception as e:
        st.error(f"Error loading dictionary: {e}")
        return {}

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('speech_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except:
        return None, None

dictionary = load_dictionary()
model, label_encoder = load_model()

# Feature extraction function
def extract_features(audio_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to process audio and make prediction
def process_audio(audio_bytes):
    """Process audio data and return prediction"""
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running train_model.py")
        return

    # Save audio to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file.write(audio_bytes)
    temp_file_path = temp_file.name
    temp_file.close()

    # Extract features and predict
    features = extract_features(temp_file_path)

    if features is not None:
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_word = label_encoder.inverse_transform(predicted_class)[0]
        confidence = np.max(prediction) * 100

        st.success(f"### Predicted English Word: **{predicted_word.upper()}**")
        st.info(f"Confidence: {confidence:.2f}%")

        # Lookup Spanish translation
        spanish_translation = dictionary.get(predicted_word.lower(), "Translation not found")
        st.success(f"### Spanish Translation: **{spanish_translation.upper()}**")

        # All probabilities
        st.write("#### All Predictions:")
        all_words = label_encoder.classes_
        probs = prediction[0]
        results = sorted(zip(all_words, probs), key=lambda x: x[1], reverse=True)

        for word, prob in results:
            spanish = dictionary.get(word.lower(), "N/A")
            st.write(f"- {word} ({spanish}): {prob*100:.2f}%")

    # Cleanup
    try:
        os.remove(temp_file_path)
    except (PermissionError, FileNotFoundError):
        pass

col1, col2 = st.columns(2)

with col1:
    st.write("### 🎙️ Record Audio")
    audio_input = st.audio_input("Click to record your voice")

    if audio_input is not None:
        st.audio(audio_input)
        process_audio(audio_input.getvalue())

with col2:
    st.write("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader("Or upload a .wav file", type=['wav'])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        process_audio(uploaded_file.read())


st.markdown("---")
st.write("### About")
st.write("This app uses a trained CNN model to recognize English spoken words and translate them to Spanish.")
st.write("**How it works:**")
st.write("1. The model extracts MFCC features from your audio")
st.write("2. The CNN classifies the English word (house, happy, learn, stop, or three)")
st.write("3. The app looks up the Spanish translation from dictionary.csv")
st.write("4. You get the translated word: casa, feliz, aprender, detener, or tres")
