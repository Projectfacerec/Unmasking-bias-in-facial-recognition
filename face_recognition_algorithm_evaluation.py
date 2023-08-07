
def load_encodings(encodings_file):
    with open(encodings_file, 'rb') as file:
        data = pickle.load(file)
    return data['encodings'], data['names']
	
def draw_rec(images_dict, unknown_image_path, encodings_file, tolerance=0.9):
    try:
        # Try to load the encodings from file
        known_face_encodings, known_face_names = load_encodings(encodings_file)
    except FileNotFoundError:
        print("No encoded file found")
        return

    # Load the unknown image
    unknown_image = face_recognition.load_image_file(unknown_image_path)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with all the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0)
        matched_names = [name for match, name in zip(matches, known_face_names) if any(match)]

        # Determine the name for the face based on the matches
        if matched_names:
            name = ', '.join(matched_names)
            print(name)
        else:
            name = "Unknown"

        # Draw a box around the face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with the name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory
    del draw

    # Display the resulting image
    plt.imshow(pil_image)
    plt.show()

draw_rec(images_dict, "link to the cohort to evaluate", encodings_file)
	
	