import os
import cv2
import numpy as np

# Path to the directory containing the images
dataset_dir = "CNN/BAYBAYIN-A"
output_dir = "CNN/white_background"

# Iterate through each image in the dataset directory
for image_file in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, image_file)
    
    # Ensure that the item in the directory is a file
    if os.path.isfile(image_path):
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        
        # Create a mask of the background
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Create a white background image
        white_background = 255 * np.ones_like(image, dtype=np.uint8)
        
        # Replace the background with white
        background_removed = cv2.bitwise_and(image, image, mask=mask_inv)
        final_image = cv2.bitwise_or(white_background, background_removed)
        
        # Save the image with white background
        output_image_path = os.path.join(output_dir, f"white_background_{image_file}")
        cv2.imwrite(output_image_path, final_image)
        print(f"Changed background to white and saved: {output_image_path}")
