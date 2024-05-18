import os
from PIL import Image

# Directory containing the JPG images
jpg_dir = "CNN/new dataset"

# Directory to save the PNG images
png_dir = "CNN/converted_images"

# Create the directory if it doesn't exist
os.makedirs(png_dir, exist_ok=True)

# Iterate through each folder in the dataset directory
for folder in os.listdir(jpg_dir):
    folder_path = os.path.join(jpg_dir, folder)
    
    # Ensure that the item in the directory is a folder
    if os.path.isdir(folder_path):
        # Create a subdirectory in the PNG directory for the current folder
        output_folder = os.path.join(png_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Iterate through each image in the folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Ensure that the item in the directory is a file and it's a JPG image
            if os.path.isfile(file_path) and file.lower().endswith('.jpg'):
                # Open the JPG image
                img = Image.open(file_path)
                
                # Convert and save as PNG format
                output_file = os.path.join(output_folder, os.path.splitext(file)[0] + '.png')
                img.save(output_file, 'PNG')
                print(f"Converted {file} to PNG and saved as {output_file}")
