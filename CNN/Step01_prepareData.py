import os
import random
import shutil

splitsize = .80
categories = []

source_folder = "CNN/limited processed_images"
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

target_folder = "CNN/dataset_for_model"
