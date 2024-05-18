import os
from PIL import Image

# Path to the directory containing the new dataset
dataset_dir = "CNN/resized_images/"

# Iterate through each folder in the dataset directory
for plant_folder in os.listdir(dataset_dir):
    plant_folder_path = os.path.join(dataset_dir, plant_folder)
    
    # Ensure that the item in the directory is a folder
    if os.path.isdir(plant_folder_path):
        # Iterate through each image in the plant folder
        for image_file in os.listdir(plant_folder_path):
            image_path = os.path.join(plant_folder_path, image_file)
            
            # Ensure that the item in the directory is a file
            if os.path.isfile(image_path):
                # Open the image
                im = Image.open(image_path)
                
                # Rotate the image
                for x in range(360):
                    rotated_im = im.rotate(x, expand=True)
                    # Create a directory for the rotated images if it doesn't exist
                    rotated_folder = os.path.join(dataset_dir, "rotated_images", plant_folder)
                    os.makedirs(rotated_folder, exist_ok=True)
                    # Save the rotated image to the rotated folder
                    rotated_image_path = os.path.join(rotated_folder, f"rotated_{x}_{image_file}")
                    rotated_im.save(rotated_image_path)
                    print(f"Rotated and saved: {rotated_image_path}")
