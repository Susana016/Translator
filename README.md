# Speech Recognition App

A machine learning-powered Streamlit app that recognizes spoken words from audio input.

## Features

- 🎙️ Record audio directly in the browser
- 📁 Upload audio files for prediction
- 🤖 CNN-based deep learning model
- 📊 Shows prediction confidence for all classes

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model on your audio data:
```bash
python train_model.py
```

This will:
- Load audio files from the `data/` folder
- Extract MFCC features from each audio sample
- Train a CNN model to classify the words
- Save the trained model as `speech_model.h5`

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Data Structure

Place your audio files (`.wav` format) in the `data/` folder, organized by word:

```
data/
  happy/
    sample1.wav
    sample2.wav
  house/
    sample1.wav
  learn/
    sample1.wav
  stop/
    sample1.wav
  three/
    sample1.wav
```

## How It Works

1. **Feature Extraction**: The app uses MFCC (Mel-frequency cepstral coefficients) to extract features from audio
2. **CNN Model**: A Convolutional Neural Network classifies the audio features
3. **Prediction**: The model outputs probabilities for each word class

## Model Architecture

- 3 Convolutional layers with MaxPooling and Dropout
- Fully connected layers for classification
- Softmax activation for multi-class prediction
