import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the image and preprocess it
imag = cv2.imread(os.getcwd() + '/CNN/Lagundi/5.jpg')
img_from_ar = Image.fromarray(imag, 'RGB')
resized_image = img_from_ar.resize((128, 128))
normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values between 0 and 1
test_image = np.expand_dims(normalized_image, axis=0)

# Load the Keras model
model = tf.keras.models.load_model('model4.keras')

# Run inference
output_data = model.predict(test_image)

# List of categories
CATEGORIES = ['Alugbati', 'Basil', 'Blueternate', 'Chess nut', 'Chives', 'Guava', 'Guyabano', 'Lagundi', 'Lemon', 'Malunggay', 'Mexican Mint', 'Miracle fruit', 'Passion fruit', 'Rose mary', 'Stevia', 'Taragon', 'Tawa tawa', 'tuway tuway','Wachichao', 'Wansoy']

# Print the percentage confidence for each category
for index, item in enumerate(CATEGORIES):
    confidence_percentage = output_data[0][index] * 100
    print(f'{item} : {confidence_percentage:.2f}%')

# Print the raw prediction probabilities
print(output_data)

confidence_threshold = 50

# Print the highest confidence and predicted class
highest_confidence = output_data[0][np.argmax(output_data)] * 100
predicted_class_index = np.argmax(output_data)
if highest_confidence < confidence_threshold:
    print("Highest Confidence: {:.2f}%".format(highest_confidence))
    print("Predicted Class: " + "Unknown")
else:   
    print("Highest Confidence: {:.2f}%".format(highest_confidence))
    print("Predicted Class: " + CATEGORIES[predicted_class_index])
