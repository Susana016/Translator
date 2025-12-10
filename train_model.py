import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Configuration
DATA_PATH = "data"
SAMPLE_RATE = 22050
MAX_PAD_LEN = 174

def extract_features(file_path):
    """Extract MFCC features from audio file"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = MAX_PAD_LEN - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data():
    """Load audio data from data folder"""
    features = []
    labels = []
    
    # Get all word folders
    word_folders = [f for f in os.listdir(DATA_PATH) 
                   if os.path.isdir(os.path.join(DATA_PATH, f))]
    
    print(f"Found word categories: {word_folders}")
    
    for word in word_folders:
        word_path = os.path.join(DATA_PATH, word)
        audio_files = [f for f in os.listdir(word_path) if f.endswith('.wav')]
        
        print(f"Processing {len(audio_files)} files for word: {word}")
        
        for audio_file in audio_files:
            file_path = os.path.join(word_path, audio_file)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(word)
    
    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    """Create CNN model for audio classification"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Loading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found! Please ensure your data folder contains audio files.")
        return
    
    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Reshape for CNN (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    input_shape = (X.shape[1], X.shape[2], 1)
    num_classes = len(label_encoder.classes_)
    
    print(f"Creating model with input shape: {input_shape}")
    model = create_model(input_shape, num_classes)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model and label encoder
    model.save('speech_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\nModel saved as 'speech_model.h5'")
    print("Label encoder saved as 'label_encoder.pkl'")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
