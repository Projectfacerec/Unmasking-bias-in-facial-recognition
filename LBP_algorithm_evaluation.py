import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Path to the dataset folder in your Google Drive
folders = {
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

for folder_name, dataset_folder in folders.items():

    # LBP parameters
    radius = 3
    n_points = 8 * radius

    def load_dataset(folder_path):
        images = []
        labels = []

        # Iterate over images in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if not os.path.isfile(image_path):
                continue

            try:
                # Read and resize the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Unable to read image: {image_path}")
                    continue

                image = cv2.resize(image, (100, 100))

                images.append(image)
                labels.append(image_name)
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(str(e))

        return images, labels

    # Load the dataset
    images, labels = load_dataset(dataset_folder)

    print('Number of images:', len(images))
    print('Number of labels:', len(labels))

    # Convert the images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Extract LBP features for the images
    lbp_features = []
    for image in images:
        lbp_image = local_binary_pattern(image, n_points, radius)
        lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(256))
        lbp_features.append(lbp_histogram)

    # Convert the LBP features to numpy array
    lbp_features = np.array(lbp_features)

    # Function to recognize faces
    def recognize_face(image):
        # Preprocess the input image
        image = cv2.resize(image, (100, 100))
        lbp_image = local_binary_pattern(image, n_points, radius)
        lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(256))

        # Calculate the Euclidean distance between the input histogram and the training histograms
        distances = np.linalg.norm(lbp_features - lbp_histogram, axis=1)

        # Find the index of the closest match
        closest_index = np.argmin(distances)

        return labels[closest_index], distances

    # Evaluation
    total_faces = len(images)
    identified_faces = 0
    unknown_faces = 0
    confidence_levels = []

    for i in range(total_faces):
        recognized_label, distances = recognize_face(images[i])
        confidence = np.min(distances)
        confidence_levels.append(confidence)

        if recognized_label == labels[i]:
            identified_faces += 1
        else:
            unknown_faces += 1

            # Display the actual image
            #actual_image = images[i]
            #plt.figure()
            #plt.title("Actual Image")
            #plt.imshow(actual_image, cmap='gray')
            #plt.axis('off')
            #plt.show()

            # Display the recognized image
            #recognized_image = images[np.where(labels == recognized_label)[0][0]]
            #plt.figure()
            #plt.title("Recognized Image")
            #plt.imshow(recognized_image, cmap='gray')
            #plt.axis('off')
            #plt.show()

    accuracy = identified_faces / total_faces
    far = unknown_faces / total_faces
    frr = 1 - accuracy
    eer = np.mean([far, frr])
    average_confidence = np.mean(confidence_levels)
    average_top_matches = np.mean(identified_faces)
    tar = identified_faces / total_faces
    print(dataset_folder)
    print('Total number of identified faces:', identified_faces)
    print('Total number of unknown faces:', unknown_faces)
    print('Accuracy:', accuracy)
    print('False Acceptance Rate (FAR):', far)
    print('False Rejection Rate (FRR):', frr)
    print('Equal Error Rate (EER):', eer)
    print('Average Confidence Level:', average_confidence)
    print('Average Top Matches:', average_top_matches)
    print('True Acceptance Rate (TAR):', tar)
