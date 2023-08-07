import os
import shutil
from PIL import Image

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to the UTKFace folder in Google Drive
utkface_folder = '/content/drive/MyDrive/UTKFace'

# Create destination folders for different ethnic groups in Google Drive
ethnicity_folders = {
    '0': '/content/drive/MyDrive/ethnicity_White',   # White
    '1': '/content/drive/MyDrive/ethnicity_Black',   # Black
    '2': '/content/drive/MyDrive/ethnicity_Asian',   # Asian
    '3': '/content/drive/MyDrive/ethnicity_Indian',  # Indian
    '4': '/content/drive/MyDrive/ethnicity_Other',   # Other
}

# Get the list of image files in the UTKFace folder
files = os.listdir(utkface_folder)

# Iterate over each file
for file in files:
    # Split the filename to extract the ethnicity
    parts = file.split('_')
    if len(parts) < 4:
        continue
    ethnicity = parts[2]

    # Determine the destination folder based on ethnicity
    destination_folder = ethnicity_folders[ethnicity]

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    source_path = os.path.join(utkface_folder, file)

    # Load the image and check if it can be opened
    try:
        img = Image.open(source_path)
        img.load()
    except (IOError, OSError):
        print(f"Failed to load image: {source_path}")
        continue

    # Copy the file to the destination folder
    shutil.copy2(source_path, destination_folder)
