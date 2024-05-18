from keras import Model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

train_path = "C:/Users/USER/tensored-django-main/MobileNetV3/final_dataset_for_model_v3/train"
validation_path = "C:/Users/USER/tensored-django-main/MobileNetV3/final_dataset_for_model_v3/validate"

trainGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input).flow_from_directory(train_path, target_size=(224, 224), batch_size=30)

ValidGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input).flow_from_directory(validation_path, target_size=(224, 224), batch_size=30)

baseModel = MobileNetV2(weights='imagenet', include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

predictionLayer = Dense(20, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=predictionLayer)

print(model.summary())

# Freeze the base model
for layer in model.layers[:-5]:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model and store history for plotting
history = model.fit(trainGenerator, validation_data=ValidGenerator, epochs=5)

# Save the model
modelSavedPath = "C:/Users/USER/tensored-django-main/MobileNetV3/dataset_for_model/mobileNetModel5.h5"
model.save(modelSavedPath)

# Plot accuracy and loss curves
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.savefig('accuracy_loss_curve.png')
plt.show()
