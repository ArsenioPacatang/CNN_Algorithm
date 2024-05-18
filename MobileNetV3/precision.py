import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Load the saved model
path_for_saved_model = "C:/Users/USER/tensored-django-main/MobileNetV3/dataset_for_model/mobileNetModel4.h5"
model = tf.keras.models.load_model(path_for_saved_model)

# Define the path to the dataset
test_data_path = "C:/Users/USER/tensored-django-main/MobileNetV3/unseen_image/validate"

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

# Predict probabilities for the test data
y_prob = model.predict(test_generator)

# Get the true labels for the test data
y_true = test_generator.classes

# Compute precision, recall, and F1 score for each class
precision = dict()

for i in range(len(test_generator.class_indices)):
    y_pred = np.argmax(y_prob, axis=1)
    precision[i] = precision_score(y_true == i, y_pred == i) * 100
    
# Compute accuracy
accuracy = accuracy_score(y_true, y_pred) * 100

# Calculate overall average precision, recall, F1 score, and accuracy
overall_precision = np.mean(list(precision.values()))
overall_accuracy = accuracy_score(y_true, y_pred) * 100

# Plot the overall metrics
plt.figure(figsize=(8, 6))
plt.bar(['Precision', 'Accuracy'], [overall_precision, overall_accuracy], color=['blue', 'green'])
plt.ylabel('Percentage')
plt.title('Overall Evaluation Metrics')
plt.ylim(0, 100)

# Add text labels to the bars
for i, v in enumerate([overall_precision, overall_accuracy]):
    plt.text(i, v + 1, f'{v:.2f}', ha='center', va='bottom')

plt.show()
