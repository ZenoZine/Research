import requests
import tarfile
import os
from tqdm import tqdm

# Define the dataset URL and save path
url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
tar_path = "VOC2012.tar"
extract_path = "VOC2012"

# Check if the file already exists to avoid re-downloading
if not os.path.exists(tar_path):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))

    # Download with progress bar
    with open(tar_path, "wb") as file, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Downloading VOC2012"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    print("Download complete.")
else:
    print("File already downloaded.")

# Verify the downloaded file
if os.path.getsize(tar_path) < 1024 * 1024:
    print("Downloaded file seems too small. Please check your internet connection and try again.")
    exit()

# Extract the .tar file with progress bar
print("Extracting VOC2012 dataset...")
os.makedirs(extract_path, exist_ok=True)

with tarfile.open(tar_path, "r") as tar:
    members = tar.getmembers()
    with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
        for member in members:
            tar.extract(member, path=extract_path)
            pbar.update(1)

print(f"Extraction complete. Files are in '{extract_path}/VOCdevkit'.")
