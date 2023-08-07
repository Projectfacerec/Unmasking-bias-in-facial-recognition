import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to the UTKFace folder in Google Drive
utkface_folder = '/content/drive/MyDrive/UTKFace'

# Create destination folders for different genders in Google Drive
male_folder = '/content/drive/MyDrive/male_folder'
female_folder = '/content/drive/MyDrive/female_folder'

# List all files in the UTKFace folder
files = os.listdir(utkface_folder)

# Iterate over each file
for file in files:
    # Split the filename to extract the gender
    parts = file.split('_')
    if len(parts) < 3:
        continue
    gender = parts[1]

    # Determine the destination folder based on gender
    if gender == '0':
        destination_folder = male_folder
    else:
        destination_folder = female_folder

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move the file to the destination folder
    source_path = os.path.join(utkface_folder, file)
    shutil.copy2(source_path, destination_folder)
