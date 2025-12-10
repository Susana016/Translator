import streamlit as st
import numpy as np
import librosa
import pickle
import os
from tensorflow import keras
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

# Page configuration
st.set_page_config(page_title="Speech Recognition App", page_icon="🎤")

st.title("🎤 Speech Recognition App")
st.write("Record your voice and the model will predict what word you said!")

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

# Recording parameters
duration = 1  # seconds
sample_rate = 22050

col1, col2 = st.columns(2)

with col1:
    if st.button("🎙️ Record Audio", use_container_width=True):
        if model is None:
            st.error("⚠️ Model not found! Please train the model first by running train_model.py")
        else:
            with st.spinner("Recording..."):
                # Record audio
                recording = sd.rec(int(duration * sample_rate), 
                                 samplerate=sample_rate, 
                                 channels=1)
                sd.wait()
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                write(temp_file.name, sample_rate, recording)
                
                # Display audio player
                st.audio(temp_file.name)
                
                # Extract features and predict
                features = extract_features(temp_file.name)
                
                if features is not None:
                    # Reshape for model input
                    features = features.reshape(1, features.shape[0], features.shape[1], 1)
                    
                    # Make prediction
                    prediction = model.predict(features, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)
                    predicted_word = label_encoder.inverse_transform(predicted_class)[0]
                    confidence = np.max(prediction) * 100
                    
                    # Display results
                    st.success(f"### Predicted Word: **{predicted_word.upper()}**")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    # Show all probabilities
                    st.write("#### All Predictions:")
                    all_words = label_encoder.classes_
                    probs = prediction[0]
                    results = sorted(zip(all_words, probs), key=lambda x: x[1], reverse=True)
                    
                    for word, prob in results:
                        st.write(f"- {word}: {prob*100:.2f}%")
                
                # Clean up temp file
                os.unlink(temp_file.name)

with col2:
    uploaded_file = st.file_uploader("Or upload an audio file", type=['wav'])
    if uploaded_file is not None:
        if model is None:
            st.error("⚠️ Model not found! Please train the model first by running train_model.py")
        else:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            # Display audio player
            st.audio(temp_file.name)
            
            # Extract features and predict
            features = extract_features(temp_file.name)
            
            if features is not None:
                # Reshape for model input
                features = features.reshape(1, features.shape[0], features.shape[1], 1)
                
                # Make prediction
                prediction = model.predict(features, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_word = label_encoder.inverse_transform(predicted_class)[0]
                confidence = np.max(prediction) * 100
                
                # Display results
                st.success(f"### Predicted Word: **{predicted_word.upper()}**")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Show all probabilities
                st.write("#### All Predictions:")
                all_words = label_encoder.classes_
                probs = prediction[0]
                results = sorted(zip(all_words, probs), key=lambda x: x[1], reverse=True)
                
                for word, prob in results:
                    st.write(f"- {word}: {prob*100:.2f}%")
            
            # Clean up temp file
            os.unlink(temp_file.name)

st.markdown("---")
st.write("### About")
st.write("This app uses a Convolutional Neural Network (CNN) trained on audio data to recognize spoken words.")
st.write("The model extracts MFCC (Mel-frequency cepstral coefficients) features from your audio and classifies the word.")
