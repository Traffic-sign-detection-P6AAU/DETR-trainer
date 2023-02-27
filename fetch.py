import requests
import time
import base64
import xml.etree.ElementTree as ET

user = "andreashp@outlook.com-585:1h7CAWb0HppdR80j"
user_bytes = user.encode("utf-8")
base64_bytes = base64.b64encode(user_bytes)
b64 = base64_bytes.decode("utf-8")

response = requests.get("https://distribution.dataudveksler.app.vd.dk/api/dataset/61/latest/DatexII", headers={"Authorization": f"Basic {b64}"})
xml = ET.fromstring(response.text)
# root = xml.getroot()

cameralist = xml.findall("d2LogicalModel/payloadPublication/genericPublicationExtension/CctvSiteTablePublication/cctvCameraList/cctvCameraMetadataRecord")

# i = xml[0][1][3][0][1]
# xml
# cameralist = i.findall(".//cctvCameraMetadataRecord")

print(len(cameralist))

urls = []
for camera in cameralist:
    print(camera.text)
    urls.append(camera)

print(urls)
# Number of images to download
# num_images = 10000

# # Directory to save the images to
# directory = "data/images/"

# # Create the directory if it does not exist
# import os
# os.makedirs(directory, exist_ok=True)

# # Loop over the number of images to download
# for i in range(num_images):
#     # Fetch the image from the API endpoint
#     response = requests.get(urls[0])

#     # Generate a filename for the image
#     filename = f"{directory}image_{i+1}.jpg"

#     # Save the image to the specified directory
#     with open(filename, "wb") as f:
#         f.write(response.content)

#     # Wait for 20 seconds before fetching the next image
#     time.sleep(20)