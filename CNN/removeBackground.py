import os
from rembg import remove
from io import BytesIO
from PIL import Image

def remove_background(input_image_path, output_image_path):
    try:
        with open(input_image_path, "rb") as image_file:
            input_image = image_file.read()
    except FileNotFoundError:
        print(f"File not found: {input_image_path}")
        return
    
    try:
        # Remove background using rembg
        output_image_bytes = remove(input_image)
        output_image = Image.open(BytesIO(output_image_bytes))
    except Exception as e:
        print(f"Error removing background: {e}")
        return
    
    try:
        # Create a new image with white background
        white_background = Image.new("RGB", output_image.size, "WHITE")
        
        # Paste the foreground (image with removed background) onto the white background
        white_background.paste(output_image, (0, 0), output_image)
        
        # Save the resulting image with white background
        white_background.save(output_image_path)
        print(f"Background removed and white background inserted: {output_image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def process_images_in_directory(input_dir, output_dir):
    # Iterate through each folder in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output subfolder structure in the output directory
        output_subfolder = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Iterate through each file in the current folder
        for file in files:
            input_image_path = os.path.join(root, file)
            output_image_path = os.path.join(output_subfolder, file)
            remove_background(input_image_path, output_image_path)

# Specify the input and output directories
input_directory = "CNN/limited dataset"
output_directory = "CNN/test_white_background"

# Process images in the input directory and save the processed images in the output directory
process_images_in_directory(input_directory, output_directory)
