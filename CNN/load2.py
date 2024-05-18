import os
import cv2
from PIL import Image
import numpy as np

class DataLoader:
    def __init__(self, directory_map):
        self.directory_map = directory_map
        self.data = []
        self.labels = []
    
    def load_data(self):
        for label, directory in self.directory_map.items():
            self.load_images(label, directory)
        
    def load_images(self, label, directory):
        dir_path = os.path.join(os.getcwd(), "CNN/data", directory)
        images = os.listdir(dir_path)
        for image_name in images:
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            resized_image = self.preprocess_image(image)
            self.data.append(np.array(resized_image))
            self.labels.append(label)
    
    def preprocess_image(self, image):
        img_from_ar = Image.fromarray(image, 'RGB')
        resized_image = img_from_ar.resize((128, 128))
        return resized_image
    
    def save_data(self):
        plants = np.array(self.data)
        labels = np.array(self.labels)
        np.save("plants4", plants)
        np.save("labels4", labels)

if __name__ == "__main__":
    directory_map = {
        0: "Alugbati",
        1: "Basil",
        2: "Blueternate",
        3: "Chess nut",
        4: "Chives",
        5: "Guava",
        6: "Guyabano",
        7: "Lagundi",
        8: "Lemon",
        9: "Malunggay",
        10: "Mexican Mint",
        11: "Miracle Fruit",
        12: "Passion fruit",
        13: "Rose mary",
        14: "Stevia",
        15: "Taragon",
        16: "Tawatawa",
        17: "Tuwaytuway",
        18: "Wachichao",
        19: "Wansoy"
    }

    data_loader = DataLoader(directory_map)
    data_loader.load_data()
    data_loader.save_data()