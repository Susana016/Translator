# English to Spanish Speech Translator

A machine learning-powered Streamlit web application that recognizes English spoken words and translates them to Spanish using a trained CNN model and dictionary-based translation.

## Features

- 🎙️ **Browser-based Audio Recording** - Record directly in your web browser using Streamlit's native audio input
- 📁 **File Upload Support** - Upload pre-recorded .wav files for translation
- 🤖 **CNN Deep Learning Model** - Convolutional Neural Network trained on MFCC audio features
- 🌐 **Dictionary Translation** - Automatic English-to-Spanish lookup via dictionary.csv
- 📊 **Confidence Metrics** - See prediction confidence scores for all word classes
- 📱 **TFLite Conversion** - Export model for mobile and edge deployment
- ✅ **WSL Compatible** - Works seamlessly in Windows Subsystem for Linux

## Supported Words

| English | Spanish  |
|---------|----------|
| happy   | feliz    |
| house   | casa     |
| learn   | aprender |
| stop    | detener  |
| three   | tres     |

## Prerequisites

- Python 3.7+
- Trained model files: `speech_model.h5` and `label_encoder.pkl`
- Audio training data organized by word category

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- numpy
- librosa
- resampy
- scikit-learn
- tensorflow
- scipy

### 2. Prepare Training Data

Organize your audio files in the following structure:

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

### 3. Train the Model

```bash
python train_model.py
```

This script will:
- Load audio files from the `data/` folder
- Extract MFCC (Mel-frequency cepstral coefficients) features
- Train a CNN model to classify the spoken words
- Save the trained model as `speech_model.h5`
- Save the label encoder as `label_encoder.pkl`

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

### Recording Audio
1. Open the application in your web browser
2. Click the microphone icon under "Record Audio"
3. Allow microphone permissions in your browser
4. Speak one of the supported English words clearly
5. The app will display:
   - The predicted English word
   - Confidence percentage
   - Spanish translation
   - All prediction probabilities

### Uploading Audio Files
1. Click "Browse files" under "Upload Audio File"
2. Select a .wav file from your computer
3. View the prediction and translation results

## How It Works

### 1. Audio Input
The application accepts audio through two methods:
- **Browser Recording**: Uses Streamlit's `st.audio_input()` for real-time microphone capture
- **File Upload**: Accepts pre-recorded .wav files

### 2. Feature Extraction
Audio is processed using MFCC (Mel-frequency cepstral coefficients):
- 40 MFCC coefficients extracted per audio sample
- Audio normalized to 22,050 Hz sample rate
- Features padded/truncated to 174 time steps for consistency

### 3. CNN Classification
A Convolutional Neural Network processes the MFCC features:
- **Layer 1**: 32 filters, 3x3 kernel, ReLU activation, MaxPooling, 25% Dropout
- **Layer 2**: 64 filters, 3x3 kernel, ReLU activation, MaxPooling, 25% Dropout
- **Layer 3**: 128 filters, 3x3 kernel, ReLU activation, MaxPooling, 25% Dropout
- **Dense Layer**: 128 neurons, ReLU activation, 50% Dropout
- **Output Layer**: 5 neurons (one per word), Softmax activation

### 4. Dictionary Translation
The predicted English word is matched against `dictionary.csv`:
- Case-insensitive matching
- Returns corresponding Spanish translation
- Displays "Translation not found" for unknown words

### 5. Results Display
The interface shows:
- Primary prediction with confidence score
- Spanish translation
- Ranked list of all predictions with probabilities and translations

## Model Architecture Details

```
Input Shape: (40, 174, 1) - MFCC features
Total Parameters: ~1.5M trainable parameters
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Training Metrics: Accuracy
```

## Dictionary Management

The translation dictionary is stored in `dictionary.csv`:

```csv
English,Spanish
happy,feliz
house,casa
learn,aprender
stop,detener
three,tres
```

To add new words:
1. Edit `dictionary.csv` to add English-Spanish pairs
2. Collect audio samples for the new English words
3. Organize samples in `data/[word_name]/` folders
4. Retrain the model with `python train_model.py`

## TFLite Model Conversion

Convert the trained model to TensorFlow Lite format for deployment on mobile devices and edge hardware:

```bash
python convert_to_tflite.py
```

**Output**: `speech_model.tflite`

**Benefits**:
- Reduced model size (typically 60-80% smaller)
- Optimized for mobile inference
- Compatible with TensorFlow Lite runtime
- Suitable for Android, iOS, Raspberry Pi, and embedded systems

**Use Cases**:
- Mobile translation apps
- Offline speech recognition
- IoT voice-controlled devices
- Edge AI applications

## File Structure

```
translator/
├── app.py                    # Main Streamlit application
├── train_model.py            # Model training script
├── convert_to_tflite.py      # TFLite conversion utility
├── dictionary.csv            # English-Spanish translation pairs
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── speech_model.h5           # Trained Keras model (generated)
├── label_encoder.pkl         # Label encoder (generated)
├── speech_model.tflite       # TFLite model (generated)
└── data/                     # Training audio data
    ├── happy/
    ├── house/
    ├── learn/
    ├── stop/
    └── three/
```

## Troubleshooting

### Model Not Found Error
- Ensure `speech_model.h5` and `label_encoder.pkl` exist
- Run `python train_model.py` to generate model files

### Audio Format Issues
- Only .wav files are supported
- Ensure audio is mono (single channel)
- Recommended: 22,050 Hz sample rate

### Low Prediction Accuracy
- Speak clearly and avoid background noise
- Record at least 100+ samples per word for training
- Ensure consistent audio quality across samples

### WSL Audio Issues
- This app uses browser-based recording via Streamlit
- No system-level audio drivers required
- Works seamlessly in WSL environments

## Performance Considerations

- **Model Loading**: Cached using `@st.cache_resource` for faster subsequent runs
- **Inference Time**: Typically 100-300ms per prediction
- **Memory Usage**: ~2GB RAM (includes TensorFlow overhead)

## Future Enhancements

Potential improvements for the project:
- Support for additional languages
- Expanded vocabulary (more words)
- Real-time continuous speech recognition
- Phrase translation (multiple words)
- User feedback mechanism for model improvement
