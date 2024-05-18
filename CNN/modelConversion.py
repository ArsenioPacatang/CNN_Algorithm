import tensorflow as tf

# Load the TensorFlow model from .h5 file
model_path = 'keras_model.h5'  # Adjust the path to your .h5 file
model = tf.keras.models.load_model(model_path)

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a .tflite file
tflite_model_path = 'model5.tflite'  # Adjust the path to save the .tflite file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved to:", tflite_model_path)
