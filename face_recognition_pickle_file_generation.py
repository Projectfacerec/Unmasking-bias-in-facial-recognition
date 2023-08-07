import os

#change this to the link of the intended cohort
folder = '/content/drive/MyDrive/ethnicity_Black'

# Create a dictionary to store the image file paths
images_dict = {}

# List all files in the folder
files = os.listdir(folder)

# Iterate over each file
for file in files:
    # Extract the file name without the extension
    file_name = os.path.splitext(file)[0]

    # Construct the file path
    file_path = os.path.join(folder, file)

    # Add the file path to the dictionary with the file name as the key
    images_dict[file_name] = [file_path]

# Print the updated images dictionary
print(images_dict)
#change this to match the name of the intended cohort
encodings_file = "/content/drive/MyDrive/ethnicity_Black/face_encodings.pickle"

def save_encodings(images_dict, encodings_file, folder):
    print(encodings_file)
    known_face_encodings = []
    known_face_names = []

    # Load and encode the known face images
    for name, image_paths in images_dict.items():
        face_encodings = []
        for image_path in image_paths:
            print(image_path)
            if image_path=="/content/drive/MyDrive/ethnicity_Other/face_encodings.pickle":
               image_page="/content/drive/MyDrive/ethnicity_Other/7_1_4_20161221193134222.jpg.chip.jpg"
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            # Check if at least one face is detected
            if len(face_locations) > 0:
                print("yes 1");
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                face_encodings.append(face_encoding)
            # Append the face encodings only if at least one face is detected
            if len(face_encodings) > 0:
                print("yes 2");
                known_face_encodings.append(face_encodings)
                known_face_names.append(name)
            else:
               os.remove(image_path)

    # Save the encodings to a file
    data = {
        'encodings': known_face_encodings,
        'names': known_face_names
    }
    with open(encodings_file, 'wb') as file:
        pickle.dump(data, file)
save_encodings(images_dict, encodings_file, folder)