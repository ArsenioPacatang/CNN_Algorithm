import os
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2 ,  preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# get the list of classes
categories = os.listdir("C:/Users/USER/tensored-django-main/MobileNetV3/dataset_for_model/train")
categories.sort()
print(categories)

# load the model
path_for_saved_model = "C:/Users/USER/tensored-django-main/MobileNetV3/dataset_for_model/mobileNetModel.h5"
model = tf.keras.models.load_model(path_for_saved_model)

# print(model.summary())

def classify_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.BILINEAR)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)

    pred = model.predict(x)
    categoryValue = np.argmax(pred, axis=1)
    print(categoryValue)

    categoryValue = categoryValue[0]
    print(categoryValue)

    result= categories[categoryValue]

    return result

imagePath = "C:/Users/USER/tensored-django-main/MobileNetV3/testing/american mint.jpg"
resultText = classify_image(imagePath)
print(resultText)


