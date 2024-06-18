import requests

# URL of the pre-traiend-model with results to download
url = "https://figshare.com/ndownloader/files/47121310"


# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a file in binary write mode
    with open("pre-trained_models.zip", "wb") as file:
        # Write the content of the response to the file
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download file. Status code:", response.status_code)

# URL of the data folder to download
url = "https://figshare.com/ndownloader/files/47121292"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a file in binary write mode
    with open("datasets.zip", "wb") as file:
        # Write the content of the response to the file
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download file. Status code:", response.status_code)

