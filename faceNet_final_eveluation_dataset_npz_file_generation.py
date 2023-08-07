
import time
import mtcnn
import numpy as np
import os
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model

#change this to the link to the folder of the intended cohort
collection_dir = '/content/drive/MyDrive/augmented_blacktest_classed2'
def extract_face(filename, required_size=(160,160)):
    detector = MTCNN()
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    faces = detector.detect_faces(pixels)

    face_array = None
    if len(faces) > 0:
      initialX, initialY, initialWidth, initialHeight = faces[0]['box']
      used_index = 0
      highestWidth = initialX
      highestHeight = initialY
      index = 0
      for face in faces:
      #With Assumption that we're going to close up the targetted person
          x, y, width, height = face['box']
          if width > highestWidth or height > highestHeight :
              used_index = index
              break

          index = index + 1

      x1, y1, width, height = faces[used_index]['box']
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      face_boundary = pixels[y1:y2, x1:x2]
      face_image = Image.fromarray(face_boundary)
      face_image = face_image.resize(required_size)
      face_array = asarray(face_image)
      # print(used_index)
      return face_array
    else:
      print(f'{filename} face cannot be detected')

    return face_array

def load_faces(directory):
    faces = []
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      face = extract_face(path)
      if face is not None:
        faces.append(face)
    return faces

def load_dataset(directory):
    x, y = [], []
    counter = 1
    start_time = time.time()
    for subdir in os.listdir(directory):
      mid_time = time.time()
      path = os.path.join(directory, subdir + '/') # e.g: train/ben_afflect/
      print(f'processing {path} ({counter} out of {len(os.listdir(directory))})...')
      faces = load_faces(path)
      labels = [subdir for i in range(len(faces))] # assign label to each face
      x.extend(faces)
      y.extend(labels)
      print('>loaded %d examples for class %s in %d seconds.' % (len(faces), subdir, time.time() - mid_time))
      counter += 1
    end_time = time.time()
    print(f'Finish processing {directory} for {end_time - start_time} seconds.')
    return asarray(x), asarray(y)

train_dir = os.path.join(collection_dir, 'train')
train_X, train_y = load_dataset(train_dir)
print(train_X.shape, train_y.shape)
val_dir = os.path.join(collection_dir, 'val')
val_X, val_y = load_dataset(val_dir)
print(val_X.shape, val_y.shape)
np.savez_compressed('blacktest22-only-dataset.npz', train_X, train_y, val_X, val_y)