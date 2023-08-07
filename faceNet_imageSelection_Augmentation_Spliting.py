import os
import shutil
from PIL import Image
from torchvision import transforms

# Define the source folders
source_folders = {
    "dataset1": "/content/drive/MyDrive/child",
    "dataset2": "/content/drive/MyDrive/ethnicity_Asian",
    "dataset3": "/content/drive/MyDrive/ethnicity_Black",
    "dataset4": "/content/drive/MyDrive/ethnicity_Other",
    "dataset5": "/content/drive/MyDrive/ethnicity_White",
    "dataset6": "/content/drive/MyDrive/ethnicity_Indian",
    "dataset7": "/content/drive/MyDrive/male_folder",
    "dataset8": "/content/drive/MyDrive/female_folder",
    "dataset9": "/content/drive/MyDrive/mid_age",
    "dataset10": "/content/drive/MyDrive/young",
    "dataset11": "/content/drive/MyDrive/old"
}

# Define the destination folder
destination_folder = "/content/drive/MyDrive/new_folders"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through each source folder
for dataset, folder_path in source_folders.items():
    # Create a new folder for the dataset
    new_folder = os.path.join(destination_folder, dataset)
    os.makedirs(new_folder)

    # Get a list of image files in the source folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Pick 3 images (or less if there are fewer than 3 images)
    selected_images = image_files[:3]

    # Copy the selected images to the new folder
    for image in selected_images:
        source_path = os.path.join(folder_path, image)
        destination_path = os.path.join(new_folder, image)
        shutil.copy(source_path, destination_path)


# Define the source folders
source_folders = {
    "child": "/content/drive/MyDrive/new_folders/child",
    "asian": "/content/drive/MyDrive/new_folders/asian",
    "black": "/content/drive/MyDrive/new_folders/black",
    "other": "/content/drive/MyDrive/new_folders/others",
    "white": "/content/drive/MyDrive/new_folders/white",
    "indian": "/content/drive/MyDrive/new_folders/indian",
    "male": "/content/drive/MyDrive/new_folders/male",
    "female": "/content/drive/MyDrive/new_folders/female",
    "mid_age": "/content/drive/MyDrive/new_folders/mid_age",
    "young": "/content/drive/MyDrive/new_folders/young",
    "old": "/content/drive/MyDrive/new_folders/old"
}

# Loop through each source folder
for folder_name, folder_path in source_folders.items():


    # Get a list of image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Print the path of each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(image_path)

    print()  # Add a blank line between folders



# Define the source folders
source_folders = {
    "child": "/content/drive/MyDrive/new_folders/child",
    "asian": "/content/drive/MyDrive/new_folders/asian",
    "black": "/content/drive/MyDrive/new_folders/black",
    "other": "/content/drive/MyDrive/new_folders/others",
    "white": "/content/drive/MyDrive/new_folders/white",
    "indian": "/content/drive/MyDrive/new_folders/indian",
    "male": "/content/drive/MyDrive/new_folders/male",
    "female": "/content/drive/MyDrive/new_folders/female",
    "mid_age": "/content/drive/MyDrive/new_folders/mid_age",
    "young": "/content/drive/MyDrive/new_folders/young",
    "old": "/content/drive/MyDrive/new_folders/old"
}

# Define the destination folder
destination_folder = "/content/drive/MyDrive/augmented_images"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Define the number of augmented images to generate per original image
num_augmented_images = 20

# Define the transformations for image augmentation
transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224),
    transforms.ToTensor()
])

# Loop through each source folder
for folder_name, folder_path in source_folders.items():
    # Get a list of image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Perform image augmentation for each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Create a new folder for the image path
        new_folder_name = image_path.replace("/", "_").replace(".", "_")
        new_folder_path = os.path.join(destination_folder, new_folder_name)
        os.makedirs(new_folder_path)

        # Open the original image
        image = Image.open(image_path)

        # Save the original image in the new folder
        original_image_path = os.path.join(new_folder_path, image_file)
        image.save(original_image_path)

        # Perform image augmentation to generate additional images
        for i in range(num_augmented_images):
            augmented_image = transform(image)
            augmented_image_path = os.path.join(new_folder_path, f"augmented_{i+1}.png")
            transforms.ToPILImage()(augmented_image).save(augmented_image_path)

def split_dataset(input_dir, output_dir, training_examples, validation_examples):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    # Iterate over subdirectories (class labels) in the input directory
    for class_label in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_label)
        if not os.path.isdir(class_dir):
            continue

        # Create subdirectories for class labels in the output directories
        train_class_dir = os.path.join(output_dir, 'train', class_label)
        val_class_dir = os.path.join(output_dir, 'val', class_label)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get list of images in the class directory
        images = os.listdir(class_dir)

        # Split the images into training and validation sets
        train_images = images[:training_examples]
        val_images = images[training_examples:training_examples+validation_examples]

        # Move training images to the corresponding output directory
        for image in train_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(train_class_dir, image)
            shutil.copy(src_path, dst_path)

        # Move validation images to the corresponding output directory
        for image in val_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(val_class_dir, image)
            shutil.copy(src_path, dst_path)
    print("done")
# Specify the input directory, output directory, and the number of training and validation examples
input_dir = '/content/drive/MyDrive/augmented_images'
output_dir = '/content/drive/MyDrive/outfolder'
training_examples = 16
validation_examples = 4

# Split the dataset
split_dataset(input_dir, output_dir, training_examples, validation_examples)
collection_dir = '/content/drive/MyDrive/outfolder'