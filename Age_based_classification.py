import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to the UTKFace folder in Google Drive
utkface_folder = '/content/drive/MyDrive/UTKFace'

# Create destination folders for different age groups in Google Drive
young = '/content/drive/MyDrive/young'
mid = '/content/drive/MyDrive/mid_age'
old = '/content/drive/MyDrive/old'
child = '/content/drive/MyDrive/child'

# List all files in the UTKFace folder
files = os.listdir(utkface_folder)

# Iterate over each file
for file in files:
    # Split the filename to extract the age
    parts = file.split('_')
    if len(parts) < 3:
        continue
    age = int(parts[0])

    # Determine the destination folder based on age
    if age >= 0 and age <= 17:
        destination_folder = child
    elif age >= 18 and age <= 30:
        destination_folder = young
    elif age >= 31 and age <= 50:
        destination_folder = mid
    else:
        destination_folder = old

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move the file to the destination folder
    source_path = os.path.join(utkface_folder, file)
    shutil.copy2(source_path, destination_folder)
