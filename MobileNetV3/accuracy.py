import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load the saved model
path_for_saved_model = "C:/Users/USER/tensored-django-main/MobileNetV3/dataset_for_model/mobileNetModel5.h5"
model = tf.keras.models.load_model(path_for_saved_model)

# Define the path to the dataset
test_data_path = "C:/Users/USER/tensored-django-main/MobileNetV3/final_dataset_for_model_v3/validate"

# Create a test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test data
result = model.evaluate(test_generator)

# Calculate accuracy
accuracy = result[1]

print("Accuracy:", accuracy)

# Plot accuracy
plt.bar(['Accuracy'], [accuracy])
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()
