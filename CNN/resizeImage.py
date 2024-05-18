from PIL import Image
import os

# Specify the folder containing the original images
original_folder = 'CNN/final dataset/'

# Specify the folder where resized images will be saved
resized_folder = 'CNN/resized_images/'

# Set the width and height for resized images
new_width = 224
new_height = 224

# Get all subfolders within the original folder
subfolders = [subfolder for subfolder in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, subfolder))]

# Loop through each subfolder
for subfolder in subfolders:
    # Create a corresponding subfolder in the resized folder
    resized_subfolder = os.path.join(resized_folder, subfolder)
    os.makedirs(resized_subfolder, exist_ok=True)

    # Get all files from the current subfolder
    files = [file for file in os.listdir(os.path.join(original_folder, subfolder)) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Loop through each file in the subfolder
    for file in files:
        # Open the image
        with Image.open(os.path.join(original_folder, subfolder, file)) as img:
            # Resize the image
            resized_img = img.resize((new_width, new_height))
            # Save the resized image to the corresponding subfolder in JPG format
            resized_img.save(os.path.join(resized_subfolder, os.path.splitext(file)[0] + '_resized.jpg'))

print("All images resized to 224x224 pixels and saved as JPG format successfully.")
