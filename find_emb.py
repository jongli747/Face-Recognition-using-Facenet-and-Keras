from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from os import listdir
from numpy import savez_compressed
from os.path import isdir
from keras.models import load_model
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from numpy import expand_dims


# extract a single face from a given photograph
def extract_face(filename, required_size = (160,160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1,y1,width,height = results[0]['box']
    #bug fix
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1 + width, y1+height
    face = pixels[y1:y2, x1:x2]
    type(face)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        #store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X,y = list(), list()
    for subdir in listdir(directory):
        #path
        path = directory + subdir + '/'
        #skip any files that might be in the dir
        if not isdir(path):
            continue
        faces = load_faces(path)
        #create label
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' %(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


def get_embedding(model, face_pixels):

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis = 0)
    #get embedding
    y0 = model.predict(samples)
#     print(y0.shape)
#     print(y0)
    return y0[0]


model = load_model("./model/facenet_keras.h5")


image, labels = load_dataset('./data/')
print(image.shape, labels.shape)

embeddings = list()
for face_pixels in image:
    emb = get_embedding(model, face_pixels)
    embeddings.append(emb)
embtrainx = np.asarray(embeddings)

print(embtrainx.shape)

np.save('face_embeddings.npy', embtrainx)
np.save('labels.npy', labels)
