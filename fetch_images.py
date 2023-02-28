import time
import requests
import base64
import xml.etree.ElementTree as ET

user = "andreashp@outlook.com-585:1h7CAWb0HppdR80j"
user_bytes = user.encode("utf-8")
base64_bytes = base64.b64encode(user_bytes)
b64 = base64_bytes.decode("utf-8")

response = requests.get("https://distribution.dataudveksler.app.vd.dk/api/dataset/61/latest/DatexII", headers={"Authorization": f"Basic {b64}"})
xml = ET.fromstring(response.text)

cameralist = xml[1][3][0][1]
start = time.time()
j = 1
k = 1
if (k < 10000):
    for cctvCameraMetadataRecord in cameralist:
        for elem in cctvCameraMetadataRecord:
            if elem.tag == "{http://datex2.eu/schema/2/2_0}cctvStillImageService":
                    url = elem[2][0].text
                    print("url: " + url)

                    # Number of images to download from each camera
                    num_images = 20

                    # Directory to save the images to
                    directory = "data/images/"

                    # Loop over the number of images to download
                    for i in range(num_images):
                        # Fetch the image from the API endpoint
                        response = requests.get(url)

                        # Generate a filename for the image
                        filename = f"{directory}camera_{j}_image_{k}.jpg"

                        # Save the image to the specified directory
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        # Wait for 15 seconds before fetching the next image
                        end = time.time()
                        print("Images downloaded: ", k)
                        print(f'Total time elapsed: {round((end-start)/60, 2)} minutes')
                        k += 1
                        time.sleep(15)
                    j += 1