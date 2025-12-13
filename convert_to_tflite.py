"""
Convert the trained Keras model to TensorFlow Lite format
"""
import tensorflow as tf
from tensorflow import keras
import os

def convert_to_tflite():
    """Convert speech_model.h5 to TFLite format"""

    # Check if model exists
    if not os.path.exists('speech_model.h5'):
        print("Error: speech_model.h5 not found!")
        print("Please train the model first using: python train_model.py")
        return

    print("Loading Keras model...")
    model = keras.models.load_model('speech_model.h5')

    print("\nModel summary:")
    model.summary()

    # Convert to TFLite
    print("\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the TFLite model
    output_file = 'speech_model.tflite'
    with open(output_file, 'wb') as f:
        f.write(tflite_model)

    # Get file sizes for comparison
    h5_size = os.path.getsize('speech_model.h5') / 1024  # KB
    tflite_size = os.path.getsize(output_file) / 1024  # KB

    print(f"\nConversion successful!")
    print(f"Original model size: {h5_size:.2f} KB")
    print(f"TFLite model size: {tflite_size:.2f} KB")
    print(f"Size reduction: {((h5_size - tflite_size) / h5_size * 100):.2f}%")
    print(f"\nTFLite model saved as: {output_file}")
    print("\nYou can now use this model for mobile deployment or edge devices!")

if __name__ == "__main__":
    convert_to_tflite()
