import os
from PIL import Image
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

# Define the list of classes
categories = ['Alugbati', 
            'American Mint', 'Tuwaytuway', 
            'Blueternate', 'Chess nut', 'Chilli Pepper', 
            'Passion Fruit', 'Guava',
            'Guyabano', 'Lagundi', 'Lemon', 
            'Malunggay', 'Mexican Mint', 'Miracle Fruit', 
            'Stevia', 'Sweet Basil', 'Taragon', 'Tawatawa', 
            'Wansoy', 'Wachichao']

# Load the pre-trained MobileNetV2 model
model_path = os.path.join(settings.BASE_DIR, 'mobileNetModel3.h5')
model = tf.keras.models.load_model(model_path)

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

@csrf_exempt
def index(request):
    try:
        # Initialize variables
        predictions = []

        # Save the uploaded image
        image_file = request.FILES["image"]
        fs = CustomFileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        path = str(settings.MEDIA_ROOT) + "/" + image_file.name

        # Read the image
        img = Image.open(path)
        img = img.resize((224, 224), Image.BILINEAR)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict the class probabilities
        pred = model.predict(x)[0]
        top_index = np.argmax(pred)
        top_prediction = categories[top_index]
        top_prediction_confidence = round(pred[top_index] * 100, 2)

        # Get the top predictions and their probabilities
        for index, confidence in enumerate(pred):
            if index != top_index:
                category = categories[index]
                confidence = round(confidence * 100, 2)
                predictions.append({"class": category, "confidence": confidence})

        # Delete the uploaded image
        fs.delete(filename)

        # Return the top predictions along with the top prediction and its confidence
        return JsonResponse({
            "prediction": top_prediction,
            "confidence": top_prediction_confidence,
            "all_predictions": predictions, 
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
