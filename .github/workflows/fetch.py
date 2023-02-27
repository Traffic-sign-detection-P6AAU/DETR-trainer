import requests
import time

# URL of the API endpoint to fetch the images from
url = "https://example.com/api/images/"

# Number of images to download
num_images = 10000

# Directory to save the images to
directory = "data/images/"

# Create the directory if it does not exist
import os
os.makedirs(directory, exist_ok=True)

# Loop over the number of images to download
for i in range(num_images):
    # Fetch the image from the API endpoint
    response = requests.get(url)

    # Generate a filename for the image
    filename = f"{directory}image_{i+1}.jpg"

    # Save the image to the specified directory
    with open(filename, "wb") as f:
        f.write(response.content)

    # Wait for 20 seconds before fetching the next image
    time.sleep(20)