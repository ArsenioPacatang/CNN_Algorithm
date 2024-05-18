import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load your data and preprocess it as you did in your code
plants = np.load("plants4.npy")
labels = np.load("labels4.npy")
s = np.arange(plants.shape[0])
np.random.shuffle(s)
plants = plants[s]
labels = labels[s]
num_classes = len(np.unique(labels))
data_length = len(plants)
(x_train, x_test) = plants[int(0.2 * data_length):], plants[:int(0.2 * data_length)]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
(y_train, y_test) = labels[int(0.2 * data_length):], labels[:int(0.2 * data_length)]

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Early Stopping if accuracy reaches 80%
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.8:
            print("\nReached 80% accuracy, so stopping training!")
            self.model.stop_training = True

# Train the model without data augmentation
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[CustomCallback()]
)

# Save the model
model.save("model4.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("model4.tflite", "wb") as f:
    f.write(tflite_model)

# Make predictions
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["class_{}".format(i) for i in range(num_classes)]))

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
