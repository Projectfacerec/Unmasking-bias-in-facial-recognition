import os
import cv2
import numpy as np

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

# Perform face recognition for each folder
for folder_name, dataset_folder in folders.items():
    print("Processing", folder_name)

    # Load the dataset
    images, labels = load_dataset(dataset_folder)

    print('Number of images:', len(images))
    print('Number of labels:', len(labels))

    # Convert the images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Flatten the images into 1D vectors
    image_vectors = images.reshape(images.shape[0], -1)

    # Compute the mean face
    mean_face = np.mean(image_vectors, axis=0)

    # Compute the difference vectors
    diff_vectors = image_vectors - mean_face

    # Compute the covariance matrix
    covariance_matrix = np.cov(diff_vectors.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Choose the top k eigenvectors (eigenfaces)
    k = 10
    eigenfaces = eigenvectors[:, :k]

    # Compute the weights for the training images
    weights = np.dot(eigenfaces.T, diff_vectors.T)

    # Function to recognize faces
    def recognize_face(image):
        # Preprocess the input image
        image = cv2.resize(image, (100, 100))
        image_vector = image.reshape(1, -1)

        # Compute the difference vector
        diff_vector = image_vector - mean_face

        # Compute the weights for the input image
        input_weights = np.dot(eigenfaces.T, diff_vector.T)

        # Calculate the Euclidean distance between the input weights and the training weights
        distances = np.linalg.norm(weights - input_weights, axis=0)

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

    accuracy = identified_faces / total_faces
    far = unknown_faces / total_faces
    frr = 1 - accuracy
    eer = np.mean([far, frr])
    average_confidence = np.mean(confidence_levels)
    average_top_matches = np.mean(identified_faces)
    tar = identified_faces / total_faces

    print('Total number of identified faces:', identified_faces)
    print('Total number of unknown faces:', unknown_faces)
    print('Accuracy:', accuracy)
    print('False Acceptance Rate (FAR):', far)
    print('False Rejection Rate (FRR):', frr)
    print('Equal Error Rate (EER):', eer)
    print('Average Confidence Level:', average_confidence)
    print('Average Top Matches:', average_top_matches)
    print('True Acceptance Rate (TAR):', tar)
    print()
