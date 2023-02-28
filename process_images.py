import cv2
import os
import numpy as np

def is_mostly_blue(img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the image to get blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate the percentage of blue pixels in the image
    blue_pixels = np.count_nonzero(mask == 255)
    total_pixels = img.shape[0] * img.shape[1]
    blue_percentage = (blue_pixels / total_pixels) * 100

    # Return True if more than 70% of the image is blue
    return blue_percentage > 70

# Define the path to the directory containing the images
path = 'data/images/'

# Iterate over the images in the directory
for filename in os.listdir(path):
    # Load the image
    img = cv2.imread(os.path.join(path, filename))

    # Check if the image is mostly blue
    if is_mostly_blue(img):
        # If it is, delete the image
        os.remove(os.path.join(path, filename))
